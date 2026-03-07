"""
GPT-2 Large Pipeline Parallelism (Case B) — PyTorch RPC
Multi-node pipeline: Stage 1 on worker1 (GPU), Stage 2 on worker2 (GPU).
Master (rank 0) drives training; workers hold model shards.

Usage (K8s):
    # Via gpt2-pipeline-rpc.yaml — 3 pods: master + worker1 + worker2
    kubectl apply -f gpt2-pipeline-rpc.yaml

Usage (local, single machine):
    python train_gpt2_pipeline.py --master-addr localhost --rank 0 &
    python train_gpt2_pipeline.py --master-addr localhost --rank 1 &
    python train_gpt2_pipeline.py --master-addr localhost --rank 2 &
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
from torch.distributed.optim import DistributedOptimizer

import pynvml
from transformers import GPT2LMHeadModel, GPT2Tokenizer

INSTANCE_COST_PER_HOUR = 0.526  # g4dn.xlarge on-demand, us-east-1


# ---------------------------------------------------------------------------
# Pipeline stages (same split as FairScale version)
# ---------------------------------------------------------------------------

def _make_causal_mask(attention_mask, dtype, device):
    """Create causal attention mask for GPT-2."""
    batch_size, seq_len = attention_mask.shape
    causal = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype)).view(1, 1, seq_len, seq_len)
    causal = causal.expand(batch_size, 1, seq_len, seq_len)
    mask = attention_mask.view(batch_size, 1, 1, seq_len)
    causal = causal * mask
    return (1.0 - causal) * torch.finfo(dtype).min


class GPT2Shard1(nn.Module):
    """Stage 1: Embedding + first 12 transformer blocks. Runs on worker1."""

    def __init__(self, model_name: str = "gpt2-large", pad_token_id: int = 50256):
        super().__init__()
        full = GPT2LMHeadModel.from_pretrained(model_name)
        split_at = full.config.n_layer // 2
        self.device = torch.device("cuda:0")
        self.wte = full.transformer.wte.to(self.device)
        self.wpe = full.transformer.wpe.to(self.device)
        self.blocks = nn.ModuleList([b.to(self.device) for b in full.transformer.h[:split_at]])
        self.pad_token_id = pad_token_id

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        batch_size, seq_len = x.shape
        position_ids = torch.arange(seq_len, device=self.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        hidden = self.wte(x) + self.wpe(position_ids)
        attention_mask = (x != self.pad_token_id).float().to(self.device)
        causal_mask = _make_causal_mask(attention_mask, hidden.dtype, self.device)
        for block in self.blocks:
            outputs = block(hidden, attention_mask=causal_mask, use_cache=False)
            hidden = outputs[0]
        return hidden.cpu()


class GPT2Shard2(nn.Module):
    """Stage 2: Last 12 blocks + ln_f + lm_head. Runs on worker2."""

    def __init__(self, model_name: str = "gpt2-large"):
        super().__init__()
        full = GPT2LMHeadModel.from_pretrained(model_name)
        split_at = full.config.n_layer // 2
        self.device = torch.device("cuda:0")
        self.blocks = nn.ModuleList([b.to(self.device) for b in full.transformer.h[split_at:]])
        self.ln_f = full.transformer.ln_f.to(self.device)
        self.lm_head = full.lm_head.to(self.device)

    def forward(self, hidden_rref):
        hidden = hidden_rref.to_here().to(self.device)
        batch_size, seq_len, _ = hidden.shape
        attention_mask = torch.ones(batch_size, seq_len, device=self.device, dtype=hidden.dtype)
        causal_mask = _make_causal_mask(attention_mask, hidden.dtype, self.device)
        for block in self.blocks:
            outputs = block(hidden, attention_mask=causal_mask, use_cache=False)
            hidden = outputs[0]
        hidden = self.ln_f(hidden)
        return self.lm_head(hidden).cpu()


# ---------------------------------------------------------------------------
# Distributed pipeline: RRef-based forward
# ---------------------------------------------------------------------------

class DistGPT2Pipeline(nn.Module):
    """Assembles GPT2Shard1 and GPT2Shard2 across workers via RPC."""

    def __init__(self, workers, split_size, model_name="gpt2-large"):
        super().__init__()
        self.split_size = split_size
        self.p1_rref = rpc.remote(workers[0], GPT2Shard1, args=(model_name,))
        self.p2_rref = rpc.remote(workers[1], GPT2Shard2, args=(model_name,))

    def forward(self, input_ids):
        # p1.remote().forward() returns RRef[Tensor] — the RPC framework wraps the
        # return value and registers send/recv nodes for dist_autograd.
        # p2.rpc_sync().forward(rref) passes that RRef to worker2 which calls
        # to_here() — another send/recv pair tracked by dist_autograd.
        outputs = []
        for chunk in input_ids.split(self.split_size, dim=0):
            x_rref = RRef(chunk)
            hidden_rref = self.p1_rref.remote().forward(x_rref)
            logits = self.p2_rref.rpc_sync().forward(hidden_rref)
            outputs.append(logits)
        return torch.cat(outputs, dim=0)

    def parameter_rrefs(self):
        p1_params = self.p1_rref.remote().parameter_rrefs().to_here()
        p2_params = self.p2_rref.remote().parameter_rrefs().to_here()
        return list(p1_params) + list(p2_params)


# GPT2Shard1/2 need parameter_rrefs for remote optimizer
def _add_parameter_rrefs(cls):
    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]
    cls.parameter_rrefs = parameter_rrefs


_add_parameter_rrefs(GPT2Shard1)
_add_parameter_rrefs(GPT2Shard2)


def _set_module_training(module_rref, mode):
    """Set train/eval mode on a remote module."""
    module_rref.to_here().train(mode)


class ClippedAdamW(torch.optim.AdamW):
    """AdamW with gradient clipping and linear warmup built in.
    Works with DistributedOptimizer because clipping and lr scaling happen
    inside step(), after dist_autograd has populated param.grad."""

    def __init__(self, params, lr=1e-3, max_grad_norm=1.0,
                 warmup_steps=0, **kwargs):
        self._max_grad_norm = max_grad_norm
        self._base_lr = lr
        self._warmup_steps = warmup_steps
        self._step_count = 0
        super().__init__(params, lr=lr, **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        self._step_count += 1
        if self._warmup_steps > 0 and self._step_count <= self._warmup_steps:
            scale = self._step_count / self._warmup_steps
            for group in self.param_groups:
                group["lr"] = self._base_lr * scale

        all_params = []
        for group in self.param_groups:
            all_params.extend(p for p in group["params"] if p.grad is not None)
        if all_params and self._max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(all_params, self._max_grad_norm)
        return super().step(closure)


def _get_gpu_stats(device_index=0):
    """Return GPU util and memory stats (called via RPC on workers)."""
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "gpu_util_pct": util.gpu,
            "gpu_mem_used_mb": round(mem.used / 1024**2),
            "gpu_mem_total_mb": round(mem.total / 1024**2),
        }
    except Exception:
        return {"gpu_util_pct": 0, "gpu_mem_used_mb": 0, "gpu_mem_total_mb": 0}


# ---------------------------------------------------------------------------
# Master: drives training
# ---------------------------------------------------------------------------

def eval_val_loss(model, val_loader, tokenizer):
    """Compute validation loss (and perplexity) without gradients."""
    total_loss = 0.0
    n_tokens = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"]
            labels = batch["input_ids"].clone()
            labels[labels == tokenizer.pad_token_id] = -100
            logits = model(input_ids)
            shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = labels[..., 1:].contiguous().view(-1)
            loss = torch.nn.functional.cross_entropy(
                shift_logits, shift_labels, ignore_index=-100, reduction="sum"
            )
            total_loss += loss.item()
            n_tokens += (shift_labels != -100).sum().item()
    return total_loss / max(n_tokens, 1)


def run_master(args, world_size):
    from datasets import load_dataset

    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    workers = [f"worker{i+1}" for i in range(world_size - 1)]
    model = DistGPT2Pipeline(workers, args.micro_batch_size, args.model)

    # wikitext requires explicit config: load_dataset("wikitext", "wikitext-2-raw-v1")
    ds_arg = args.dataset
    if ds_arg == "wikitext":
        ds_path, ds_config = "wikitext", "wikitext-2-raw-v1"
    elif ds_arg in ("wikitext-2-raw-v1", "wikitext-2-v1", "wikitext-103-raw-v1", "wikitext-103-v1"):
        ds_path, ds_config = "wikitext", ds_arg
    else:
        ds_path, ds_config = ds_arg, None

    def _load(split):
        if ds_config is not None:
            return load_dataset(ds_path, ds_config, split=split)
        return load_dataset(ds_path, split=split)

    raw_train = _load("train")
    raw_train = raw_train.filter(lambda x: len(x.get("text", "").strip()) > 0)
    if args.train_samples and args.train_samples < len(raw_train):
        raw_train = raw_train.select(range(args.train_samples))
    raw_val = _load("validation")
    raw_val = raw_val.filter(lambda x: len(x.get("text", "").strip()) > 0)

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_len,
            padding="max_length",
        )

    tokenized_train = raw_train.map(tokenize, batched=True, remove_columns=raw_train.column_names)
    tokenized_val = raw_val.map(tokenize, batched=True, remove_columns=raw_val.column_names)
    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask"])
    tokenized_val.set_format("torch", columns=["input_ids", "attention_mask"])

    train_loader = DataLoader(
        tokenized_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        tokenized_val,
        batch_size=args.batch_size,
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else int(0.05 * total_steps)

    if args.max_grad_norm > 0:
        dist_optim = DistributedOptimizer(
            ClippedAdamW, model.parameter_rrefs(),
            lr=args.lr, max_grad_norm=args.max_grad_norm,
            warmup_steps=warmup_steps,
        )
    else:
        dist_optim = DistributedOptimizer(
            torch.optim.AdamW, model.parameter_rrefs(), lr=args.lr,
        )

    os.makedirs(args.output_dir, exist_ok=True)
    num_gpus = world_size - 1
    global_batch_size = args.batch_size

    num_micro_batches = args.batch_size // args.micro_batch_size

    print("=" * 55)
    print("  GPT-2 Pipeline Parallelism (PyTorch RPC)")
    print("=" * 55)
    print(f"  Pipeline stages:    {num_gpus}")
    print(f"  Dataset:            {args.dataset}")
    print(f"  Train samples:      {len(tokenized_train)}")
    print(f"  Val samples:        {len(tokenized_val)}")
    print(f"  Batch size:         {args.batch_size}")
    print(f"  Micro-batch size:   {args.micro_batch_size}")
    print(f"  Micro-batches/step: {num_micro_batches}")
    print(f"  Epochs:             {args.epochs}")
    print(f"  LR:                 {args.lr}")
    print(f"  Max grad norm:      {args.max_grad_norm}")
    print(f"  Warmup steps:       {warmup_steps}")
    print("")

    epoch_results = []
    step_log = []
    nan_abort = False

    for epoch in range(args.epochs):
        rpc.rpc_sync("worker1", _set_module_training, args=(model.p1_rref, True))
        rpc.rpc_sync("worker2", _set_module_training, args=(model.p2_rref, True))
        running_loss = 0.0
        n_samples = 0
        nan_count = 0
        step = 0
        forward_times = []
        backward_times = []
        optimizer_times = []
        step_times = []
        gpu_utils = []
        gpu_mems = []
        gpu1_utils = []
        gpu2_utils = []
        gpu1_mems = []
        gpu2_mems = []

        t_epoch = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"]
            labels = batch["input_ids"].clone()
            labels[labels == tokenizer.pad_token_id] = -100

            t0 = time.time()
            with dist_autograd.context() as context_id:
                logits = model(input_ids)
                shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
                shift_labels = labels[..., 1:].contiguous().view(-1)
                loss = torch.nn.functional.cross_entropy(
                    shift_logits, shift_labels, ignore_index=-100
                )
                t1 = time.time()
                dist_autograd.backward(context_id, [loss])
                t2 = time.time()
                dist_optim.step(context_id)
            t3 = time.time()

            bs = input_ids.size(0)
            loss_val = loss.item()
            if math.isnan(loss_val) or math.isinf(loss_val):
                nan_count += 1
                if nan_count >= 5:
                    print(f"\n  *** STOPPING: {nan_count} consecutive NaN/Inf losses. "
                          f"Weights are likely corrupted. ***\n", flush=True)
                    nan_abort = True
                    break
                continue
            nan_count = 0
            running_loss += loss_val * bs
            n_samples += bs

            fwd_ms = (t1 - t0) * 1000
            bwd_ms = (t2 - t1) * 1000
            opt_ms = (t3 - t2) * 1000
            total_ms = (t3 - t0) * 1000

            forward_times.append(fwd_ms)
            backward_times.append(bwd_ms)
            optimizer_times.append(opt_ms)
            step_times.append(total_ms)

            if step % args.log_every == 0:
                s1 = rpc.rpc_sync("worker1", _get_gpu_stats, args=(0,))
                s2 = rpc.rpc_sync("worker2", _get_gpu_stats, args=(0,))
                gpu1_util = s1["gpu_util_pct"]
                gpu2_util = s2["gpu_util_pct"]
                gpu1_mem = s1["gpu_mem_used_mb"]
                gpu2_mem = s2["gpu_mem_used_mb"]
                gpu_total = s1["gpu_mem_total_mb"]
                avg_util = (gpu1_util + gpu2_util) / 2
                peak_mem = max(gpu1_mem, gpu2_mem)
            else:
                gpu1_util = gpu1_utils[-1] if gpu1_utils else 0
                gpu2_util = gpu2_utils[-1] if gpu2_utils else 0
                gpu1_mem = gpu1_mems[-1] if gpu1_mems else 0
                gpu2_mem = gpu2_mems[-1] if gpu2_mems else 0
                gpu_total = 15360
                avg_util = gpu_utils[-1] if gpu_utils else 0
                peak_mem = gpu_mems[-1] if gpu_mems else 0
            gpu_utils.append(avg_util)
            gpu_mems.append(peak_mem)
            gpu1_utils.append(gpu1_util)
            gpu2_utils.append(gpu2_util)
            gpu1_mems.append(gpu1_mem)
            gpu2_mems.append(gpu2_mem)

            if step % args.log_every == 0:
                throughput = bs / (total_ms / 1000) if total_ms > 0 else 0
                mem_util_pct = round(peak_mem / gpu_total * 100, 1) if gpu_total else 0
                entry = {
                    "epoch": epoch + 1,
                    "step": step,
                    "loss": round(loss.item(), 4),
                    "step_time_ms": round(total_ms, 1),
                    "forward_ms": round(fwd_ms, 1),
                    "backward_ms": round(bwd_ms, 1),
                    "optimizer_ms": round(opt_ms, 1),
                    "throughput_samples_sec": round(throughput, 1),
                    "gpu_util_pct": avg_util,
                    "gpu1_util_pct": gpu1_util,
                    "gpu2_util_pct": gpu2_util,
                    "gpu_mem_used_mb": peak_mem,
                    "gpu1_mem_used_mb": gpu1_mem,
                    "gpu2_mem_used_mb": gpu2_mem,
                    "gpu_mem_util_pct": mem_util_pct,
                }
                step_log.append(entry)
                print(f"  [Epoch {epoch+1}] Step {step:>4d}/{len(train_loader)}  "
                      f"Loss: {loss.item():.4f}  GPU: {avg_util:.0f}%  "
                      f"({gpu1_util}%|{gpu2_util}%)  Mem: {peak_mem}MB")

            step += 1

        if nan_abort:
            print("  Aborting remaining epochs due to NaN loss.", flush=True)
            break

        elapsed = time.time() - t_epoch
        if n_samples == 0:
            print("  No valid samples in this epoch (all NaN). Skipping.", flush=True)
            continue
        avg_loss = running_loss / n_samples
        throughput = n_samples / elapsed
        avg_fwd = sum(forward_times) / len(forward_times)
        avg_bwd = sum(backward_times) / len(backward_times)
        avg_opt = sum(optimizer_times) / len(optimizer_times)
        avg_step = sum(step_times) / len(step_times)
        _s1 = rpc.rpc_sync("worker1", _get_gpu_stats, args=(0,))
        gpu_total = _s1["gpu_mem_total_mb"]

        cost_epoch = (elapsed / 3600) * INSTANCE_COST_PER_HOUR * num_gpus
        cost_per_sample = cost_epoch / n_samples

        # Pipeline bubble: with rpc_sync, stages execute sequentially per
        # micro-batch. The theoretical bubble for K stages and M micro-batches
        # in an ideal async pipeline is (K-1)/(M+K-1). With sync execution,
        # each stage is idle ~50% of the forward/backward time for K=2.
        bubble_theoretical = (num_gpus - 1) / (num_micro_batches + num_gpus - 1) * 100
        bubble_sync = (num_gpus - 1) / num_gpus * 100

        avg_gpu1_util = sum(gpu1_utils) / len(gpu1_utils) if gpu1_utils else 0
        avg_gpu2_util = sum(gpu2_utils) / len(gpu2_utils) if gpu2_utils else 0

        rpc.rpc_sync("worker1", _set_module_training, args=(model.p1_rref, False))
        rpc.rpc_sync("worker2", _set_module_training, args=(model.p2_rref, False))
        val_loss = eval_val_loss(model, val_loader, tokenizer)
        val_perplexity = math.exp(min(val_loss, 20))

        print(f"\n  ── Epoch {epoch+1} results ──")
        print(f"  Wall time:       {elapsed:.1f}s")
        print(f"  Avg loss:        {avg_loss:.4f}")
        print(f"  Val loss:        {val_loss:.4f}")
        print(f"  Val perplexity:  {val_perplexity:.2f}")
        print(f"  Throughput:      {throughput:.1f} samples/sec")
        print(f"  Avg GPU util:    {sum(gpu_utils)/len(gpu_utils):.1f}%  "
              f"(stage1: {avg_gpu1_util:.1f}% | stage2: {avg_gpu2_util:.1f}%)")
        print(f"  Peak GPU mem:    {max(gpu_mems)} MB / {gpu_total} MB")
        print(f"  Avg step time:   {avg_step:.1f}ms  (fwd: {avg_fwd:.1f} | bwd: {avg_bwd:.1f} | opt: {avg_opt:.1f})")
        print(f"  Pipeline bubble: {bubble_sync:.1f}% (sync)  {bubble_theoretical:.1f}% (async ideal)")
        print(f"  Cost (epoch):    ${cost_epoch:.4f}")
        print(f"  Cost (sample):   ${cost_per_sample:.6f}")
        print("")

        epoch_record = {
            "epoch": epoch + 1,
            "dataset": args.dataset,
            "num_stages": num_gpus,
            "num_micro_batches": num_micro_batches,
            "global_batch_size": global_batch_size,
            "wall_time_sec": round(elapsed, 1),
            "avg_loss": round(avg_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_perplexity": round(val_perplexity, 2),
            "throughput_samples_sec": round(throughput, 1),
            "avg_gpu_util_pct": round(sum(gpu_utils) / len(gpu_utils), 1),
            "avg_gpu1_util_pct": round(avg_gpu1_util, 1),
            "avg_gpu2_util_pct": round(avg_gpu2_util, 1),
            "peak_gpu_mem_mb": max(gpu_mems),
            "gpu_mem_total_mb": gpu_total,
            "avg_step_time_ms": round(avg_step, 1),
            "avg_forward_ms": round(avg_fwd, 1),
            "avg_backward_ms": round(avg_bwd, 1),
            "avg_optimizer_ms": round(avg_opt, 1),
            "pipeline_bubble_pct": round(bubble_sync, 1),
            "pipeline_bubble_async_pct": round(bubble_theoretical, 1),
            "cost_usd_epoch": round(cost_epoch, 4),
            "cost_usd_per_sample": round(cost_per_sample, 6),
        }
        epoch_results.append(epoch_record)
        print(f"  EPOCH_JSON: {json.dumps(epoch_record)}")

    with open(os.path.join(args.output_dir, "epoch_metrics.jsonl"), "w") as f:
        for r in epoch_results:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.output_dir, "step_metrics.jsonl"), "w") as f:
        for s in step_log:
            f.write(json.dumps(s) + "\n")

    print(f"  Saved metrics to {args.output_dir}")
    print(f"    - epoch_metrics.jsonl  (per-epoch summaries)")
    print(f"    - step_metrics.jsonl   (per-step detail)")

    print("\n===== BEGIN epoch_metrics.jsonl =====", flush=True)
    for r in epoch_results:
        print(json.dumps(r), flush=True)
    print("===== END epoch_metrics.jsonl =====", flush=True)

    print("\n===== BEGIN step_metrics.jsonl =====", flush=True)
    for s in step_log:
        print(json.dumps(s), flush=True)
    print("===== END step_metrics.jsonl =====", flush=True)

    print("\nDone.", flush=True)

    if args.sleep_after > 0:
        print(f"  Sleeping {args.sleep_after}s for kubectl cp ...", flush=True)
        time.sleep(args.sleep_after)


# ---------------------------------------------------------------------------
# Worker: holds model shard, waits for RPC
# ---------------------------------------------------------------------------

def run_worker(rank):
    # Worker just needs to init RPC and wait; shards are created on demand by master
    pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-large")
    parser.add_argument("--dataset", default="wikitext",
                       help="HuggingFace dataset (wikitext uses config wikitext-2-raw-v1)")
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=2,
                       help="Micro-batch size for pipeline (split of batch)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Max gradient norm for clipping (0 = disabled)")
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="Linear warmup steps (0 = auto 5%% of total)")
    parser.add_argument("--train-samples", type=int, default=None,
                       help="Limit training set size (None = full dataset)")
    parser.add_argument("--output-dir", default="/workspace/output")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--sleep-after", type=int, default=0,
                       help="Seconds to sleep after training (keeps pod alive for kubectl cp)")
    parser.add_argument("--master-addr", default=os.environ.get("MASTER_ADDR", "localhost"))
    parser.add_argument("--master-port", default=os.environ.get("MASTER_PORT", "29500"))
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", os.environ.get("JOB_COMPLETION_INDEX", "0"))))
    parser.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "3")))
    return parser.parse_args()


def main():
    args = parse_args()
    rank = args.rank
    world_size = args.world_size

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=4, rpc_timeout=600)
    # Enable GPU tensor transfer (TensorPipe defaults to CPU-only)
    if torch.cuda.is_available():
        my_name = "master" if rank == 0 else f"worker{rank}"
        for w in ["master", "worker1", "worker2"]:
            if w != my_name:
                options.set_device_map(w, {0: 0})

    if rank == 0:
        rpc.init_rpc("master", rank=rank, world_size=world_size, rpc_backend_options=options)
        run_master(args, world_size)
    else:
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options,
        )
        run_worker(rank)

    rpc.shutdown()
    return 0


if __name__ == "__main__":
    exit(main())
