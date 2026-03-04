"""
BERT Fine-Tuning with Horovod Data Parallelism (Case A)
Distributed training benchmark on EKS with NVIDIA T4 GPUs.

Usage (via MPI Operator):
    mpirun -np 2 python train_bert.py --dataset sst2 --epochs 3
"""

import argparse
import json
import os
import time

import torch
import horovod.torch as hvd
import pynvml
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score

INSTANCE_COST_PER_HOUR = 0.526  # g4dn.xlarge on-demand, us-east-1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="sst2", choices=["sst2", "mnli"])
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Per-worker micro-batch size")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Base learning rate (auto-scaled by world size)")
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=10,
                        help="Log step-level metrics every N steps")
    parser.add_argument("--output-dir", default="/workspace/output")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg):
    """Print only from rank 0 to avoid duplicated output."""
    if hvd.rank() == 0:
        print(msg, flush=True)


def prepare_data(dataset_name, tokenizer, max_seq_len):
    """Download a GLUE task via HuggingFace and tokenize it."""

    if dataset_name == "sst2":
        raw = load_dataset("glue", "sst2")
        num_labels = 2
        train_key, val_key = "train", "validation"

        def tokenize(batch):
            return tokenizer(
                batch["sentence"],
                padding="max_length", truncation=True, max_length=max_seq_len,
            )
    else:
        raw = load_dataset("glue", "mnli")
        num_labels = 3
        train_key, val_key = "train", "validation_matched"

        def tokenize(batch):
            return tokenizer(
                batch["premise"], batch["hypothesis"],
                padding="max_length", truncation=True, max_length=max_seq_len,
            )

    drop_cols = lambda split: [c for c in raw[split].column_names if c != "label"]

    train_ds = raw[train_key].map(tokenize, batched=True,
                                  remove_columns=drop_cols(train_key))
    val_ds = raw[val_key].map(tokenize, batched=True,
                              remove_columns=drop_cols(val_key))

    train_ds.set_format("torch")
    val_ds.set_format("torch")

    return train_ds, val_ds, num_labels


def evaluate(model, loader, device):
    """Run inference on the validation set and return accuracy."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(batch["label"].tolist())

    return accuracy_score(all_labels, all_preds)


class GPUMonitor:
    """Thin wrapper around pynvml to sample GPU utilization and memory."""

    def __init__(self, device_index):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        self.total_mb = mem.total / 1024 ** 2

    def sample(self):
        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        return {
            "gpu_util_pct": util.gpu,
            "gpu_mem_used_mb": round(mem.used / 1024 ** 2),
            "gpu_mem_total_mb": round(self.total_mb),
            "gpu_mem_util_pct": round(mem.used / mem.total * 100, 1),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ── 1. Horovod init ────────────────────────────────────────────────────
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())
    device = torch.device("cuda", hvd.local_rank())
    gpu = GPUMonitor(hvd.local_rank())

    log("=" * 55)
    log("  BERT Data-Parallel Training (Horovod + NCCL)")
    log("=" * 55)
    log(f"  Workers:          {hvd.size()}")
    log(f"  Dataset:          {args.dataset.upper()}")
    log(f"  Per-worker batch: {args.batch_size}")
    log(f"  Global batch:     {args.batch_size * hvd.size()}")
    log(f"  Base LR:          {args.lr}")
    log(f"  Scaled LR:        {args.lr * hvd.size()}")
    log(f"  Epochs:           {args.epochs}")
    log(f"  GPU memory:       {gpu.total_mb:.0f} MB per device")
    log("")

    # ── 2. Data ────────────────────────────────────────────────────────────
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    log("  Loading and tokenizing dataset ...")
    train_ds, val_ds, num_labels = prepare_data(
        args.dataset, tokenizer, args.max_seq_len
    )
    log(f"  Train: {len(train_ds):,} samples")
    log(f"  Val:   {len(val_ds):,} samples")

    # DistributedSampler slices the dataset so each worker gets a unique
    # partition of size len(train_ds) // world_size.
    sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds, num_replicas=hvd.size(), rank=hvd.rank(),
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=sampler, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2,
        num_workers=2, pin_memory=True,
    )

    # ── 3. Model + optimizer ───────────────────────────────────────────────
    log("  Loading BERT-base-uncased ...\n")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=num_labels,
    ).to(device)

    # Linear LR scaling rule: when the effective batch size grows by N,
    # scale the learning rate by N to maintain the same gradient noise ratio.
    scaled_lr = args.lr * hvd.size()
    optimizer = torch.optim.AdamW(model.parameters(), lr=scaled_lr)

    # Broadcast: rank 0's weights become the single source of truth.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Wrap the optimizer so every .step() triggers an AllReduce on gradients.
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(),
    )

    # ── 4. Training loop ──────────────────────────────────────────────────
    epoch_results = []
    step_log = []

    for epoch in range(args.epochs):
        model.train()
        sampler.set_epoch(epoch)

        running_loss = 0.0
        n_samples = 0
        gpu_utils = []
        gpu_mems = []
        forward_times = []
        backward_times = []
        allreduce_times = []
        step_times = []

        t_epoch = time.time()

        for step, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # ── Phase timing ──────────────────────────────────────────
            # torch.cuda.synchronize() forces the CPU to wait for all
            # queued GPU ops to finish, giving us accurate wall-clock
            # timestamps for each phase.

            torch.cuda.synchronize()
            t0 = time.time()

            optimizer.zero_grad()
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss

            torch.cuda.synchronize()
            t1 = time.time()

            loss.backward()

            torch.cuda.synchronize()
            t2 = time.time()

            # optimizer.step() is where Horovod injects the AllReduce.
            # Horovod starts AllReduce asynchronously during backward(),
            # but .step() blocks until all gradients are synchronized.
            optimizer.step()

            torch.cuda.synchronize()
            t3 = time.time()

            # ── Bookkeeping ───────────────────────────────────────────
            bs = ids.size(0)
            running_loss += loss.item() * bs
            n_samples += bs

            fwd_ms = (t1 - t0) * 1000
            bwd_ms = (t2 - t1) * 1000
            ar_ms = (t3 - t2) * 1000
            total_ms = (t3 - t0) * 1000

            forward_times.append(fwd_ms)
            backward_times.append(bwd_ms)
            allreduce_times.append(ar_ms)
            step_times.append(total_ms)

            gpu_stats = gpu.sample()
            gpu_utils.append(gpu_stats["gpu_util_pct"])
            gpu_mems.append(gpu_stats["gpu_mem_used_mb"])

            # ── Step-level log (every N steps, rank 0 only) ───────────
            if step % args.log_every == 0 and hvd.rank() == 0:
                entry = {
                    "epoch": epoch + 1,
                    "step": step,
                    "loss": round(loss.item(), 4),
                    "step_time_ms": round(total_ms, 1),
                    "forward_ms": round(fwd_ms, 1),
                    "backward_ms": round(bwd_ms, 1),
                    "allreduce_ms": round(ar_ms, 1),
                    "throughput_samples_sec": round(
                        (bs * hvd.size()) / (total_ms / 1000), 1
                    ),
                    "gpu_util_pct": gpu_stats["gpu_util_pct"],
                    "gpu_mem_used_mb": gpu_stats["gpu_mem_used_mb"],
                    "gpu_mem_util_pct": gpu_stats["gpu_mem_util_pct"],
                }
                step_log.append(entry)

            if step % 50 == 0:
                log(f"  [Epoch {epoch+1}] Step {step:>4d}/{len(train_loader)}"
                    f"  Loss: {loss.item():.4f}"
                    f"  GPU: {gpu_stats['gpu_util_pct']}%"
                    f"  Mem: {gpu_stats['gpu_mem_used_mb']}MB")

        elapsed = time.time() - t_epoch
        avg_loss = running_loss / n_samples
        global_samples = n_samples * hvd.size()
        throughput = global_samples / elapsed

        avg_fwd = sum(forward_times) / len(forward_times)
        avg_bwd = sum(backward_times) / len(backward_times)
        avg_ar = sum(allreduce_times) / len(allreduce_times)
        avg_step = sum(step_times) / len(step_times)
        comm_overhead_pct = (avg_ar / avg_step) * 100

        cost_epoch = (elapsed / 3600) * INSTANCE_COST_PER_HOUR * hvd.size()
        cost_per_sample = cost_epoch / global_samples

        log(f"\n  ── Epoch {epoch+1} results ──")
        log(f"  Wall time:       {elapsed:.1f}s")
        log(f"  Avg loss:        {avg_loss:.4f}")
        log(f"  Throughput:      {throughput:.1f} samples/sec (global)")
        log(f"  Avg GPU util:    {sum(gpu_utils)/len(gpu_utils):.1f}%")
        log(f"  Peak GPU mem:    {max(gpu_mems)} MB / {gpu.total_mb:.0f} MB")
        log(f"  Avg step time:   {avg_step:.1f}ms"
            f"  (fwd: {avg_fwd:.1f} | bwd: {avg_bwd:.1f} | sync: {avg_ar:.1f})")
        log(f"  Comm overhead:   {comm_overhead_pct:.1f}%")
        log(f"  Cost (epoch):    ${cost_epoch:.4f}")
        log(f"  Cost (sample):   ${cost_per_sample:.6f}")

        # ── 5. Validation (rank 0 only) ───────────────────────────────────
        # Other workers idle-wait here; this is fine since validation is fast
        # relative to training, and it keeps the code simple.
        val_acc = None
        if hvd.rank() == 0:
            val_acc = evaluate(model, val_loader, device)
            log(f"  Val accuracy:    {val_acc:.4f}")
        log("")

        if hvd.rank() == 0:
            epoch_results.append({
                "epoch": epoch + 1,
                "dataset": args.dataset,
                "num_workers": hvd.size(),
                "global_batch_size": args.batch_size * hvd.size(),
                "wall_time_sec": round(elapsed, 1),
                "avg_loss": round(avg_loss, 4),
                "val_accuracy": round(val_acc, 4) if val_acc else None,
                "throughput_samples_sec": round(throughput, 1),
                "avg_gpu_util_pct": round(sum(gpu_utils) / len(gpu_utils), 1),
                "peak_gpu_mem_mb": max(gpu_mems),
                "gpu_mem_total_mb": round(gpu.total_mb),
                "avg_step_time_ms": round(avg_step, 1),
                "avg_forward_ms": round(avg_fwd, 1),
                "avg_backward_ms": round(avg_bwd, 1),
                "avg_allreduce_ms": round(avg_ar, 1),
                "communication_overhead_pct": round(comm_overhead_pct, 1),
                "cost_usd_epoch": round(cost_epoch, 4),
                "cost_usd_per_sample": round(cost_per_sample, 6),
            })

    # ── 6. Save ────────────────────────────────────────────────────────────
    if hvd.rank() == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        with open(os.path.join(args.output_dir, "epoch_metrics.jsonl"), "w") as f:
            for r in epoch_results:
                f.write(json.dumps(r) + "\n")

        with open(os.path.join(args.output_dir, "step_metrics.jsonl"), "w") as f:
            for s in step_log:
                f.write(json.dumps(s) + "\n")

        log(f"  Saved model + metrics to {args.output_dir}")
        log(f"    - epoch_metrics.jsonl  (per-epoch summaries for tables)")
        log(f"    - step_metrics.jsonl   (per-step detail for charts)")


if __name__ == "__main__":
    main()
