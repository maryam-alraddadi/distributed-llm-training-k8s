# Optimizing Distributed Training of Large Language Models on Cloud-Native Infrastructure

A complete, reproducible framework for distributed LLM training on Kubernetes, implementing and benchmarking Data Parallelism and Pipeline Model Parallelism on AWS EKS with NVIDIA T4 GPUs.

**Paper:** *Optimizing Distributed Training of Large Language Models on Cloud-Native Infrastructure* (IEEE Tutorial, HPCC Capstone 2026)

## Overview

This repository contains all source code, infrastructure configurations, Kubernetes manifests, and experimental results for the tutorial paper. Two distributed training strategies are implemented end-to-end:

| | Case A: Data Parallelism | Case B: Pipeline Parallelism |
|---|---|---|
| **Model** | BERT-Base (110M params) | GPT-2 Large (774M params) |
| **Framework** | Horovod + NCCL | PyTorch Distributed RPC |
| **Dataset** | SST-2 (67,349 samples) | WikiText-2 (23,767 samples) |
| **GPUs** | 2x NVIDIA T4 (replicated) | 2x NVIDIA T4 (partitioned) |
| **Throughput** | 84.9 samples/s | 2.7 samples/s |
| **Scaling Efficiency** | 78.3% | N/A (model partitioning) |
| **GPU Utilization** | ≥99.7% | 78% avg (94% / 62%) |
| **Cost per Epoch** | $0.23 | $2.54 |

## Repository Structure

```
.
├── infra/                          # Terraform infrastructure as code
│   ├── provider.tf                 # AWS provider, backend (S3), Helm/K8s providers
│   ├── vpc.tf                      # VPC, subnets, NAT gateway
│   ├── eks.tf                      # EKS cluster, system + GPU node groups
│   └── k8s-addons.tf               # NVIDIA device plugin, MPI Operator
│
├── training/                       # Training scripts, manifests, Dockerfiles
│   ├── train_bert.py               # Case A: BERT fine-tuning with Horovod
│   ├── train_gpt2_pipeline.py      # Case B: GPT-2 pipeline with PyTorch RPC
│   ├── bert-1gpu.yaml              # MPIJob manifest: 1-GPU baseline
│   ├── bert-2gpu.yaml              # MPIJob manifest: 2-GPU data parallel
│   ├── gpt2-pipeline-rpc.yaml      # Indexed Job + headless Service: pipeline
│   ├── Dockerfile / Dockerfile.bert # Case A Docker image (Horovod base)
│   └── Dockerfile.gpt              # Case B Docker image (PyTorch base)
│
├── results/                        # Raw experimental metrics
│   ├── 1gpu-sst2/                  # BERT 1-GPU baseline results
│   ├── 2gpu-sst2/                  # BERT 2-GPU data parallel results
│   └── 2gpu-wikitext/              # GPT-2 pipeline parallel results
│
├── main.tex                        # Tutorial paper (LaTeX source)
├── final-deliverables.tex          # Project deliverables report
└── README.md                       # This file
```

## Prerequisites

- **AWS Account** with permissions for EKS, EC2, VPC, ECR, S3
- **AWS CLI** configured (`aws configure`)
- **Terraform** ≥ 1.5
- **kubectl** configured for EKS
- **Docker** for building training images

## Tutorial Walkthrough

### Step 1: Provision Cloud Infrastructure

The entire AWS environment is defined as Terraform Infrastructure as Code. This creates a VPC with two availability zones, an EKS v1.29 cluster, a system node (t3.medium), two GPU training nodes (g4dn.xlarge with NVIDIA T4), and installs the NVIDIA device plugin and MPI Operator.

```bash
cd infra/

# Initialize Terraform (downloads providers, sets up S3 backend)
terraform init

# Preview what will be created
terraform plan

# Provision everything (takes ~15 minutes)
terraform apply
```

After provisioning, configure kubectl:

```bash
aws eks update-kubeconfig --region us-east-1 --name hpcc-capstone-cluster
```

Verify the cluster:

```bash
# All 3 nodes should be Ready
kubectl get nodes

# GPU nodes should show nvidia.com/gpu: 1
kubectl describe node -l accelerator=nvidia-t4 | grep -A5 "Allocatable"
```

### Step 2: Build and Push Docker Images

Each training case uses a dedicated Docker image pushed to Amazon ECR.

**Case A (BERT / Horovod):**

```bash
cd training/

# Create ECR repository
aws ecr create-repository --repository-name hpcc-capstone/bert-training

# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Build and push
docker build -f Dockerfile.bert -t <account-id>.dkr.ecr.us-east-1.amazonaws.com/hpcc-capstone/bert-training:v1 .
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/hpcc-capstone/bert-training:v1
```

**Case B (GPT-2 / PyTorch RPC):**

```bash
aws ecr create-repository --repository-name hpcc-capstone/gpt-pipeline-training

docker build -f Dockerfile.gpt -t <account-id>.dkr.ecr.us-east-1.amazonaws.com/hpcc-capstone/gpt-pipeline-training:v1 .
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/hpcc-capstone/gpt-pipeline-training:v1
```

> **Note:** Update the `image:` fields in the YAML manifests with your ECR URIs.

### Step 3: Run Case A — BERT Data-Parallel Training

Case A uses the Kubeflow MPI Operator to run Horovod-based data-parallel training of BERT-Base on SST-2.

**1-GPU Baseline:**

```bash
kubectl apply -f training/bert-1gpu.yaml

# Monitor training
kubectl logs -f $(kubectl get pods -l training.kubeflow.org/job-name=bert-dp-1gpu -o name | head -1)
```

**2-GPU Data Parallel:**

```bash
kubectl apply -f training/bert-2gpu.yaml

# Monitor the launcher pod
kubectl logs -f $(kubectl get pods -l training.kubeflow.org/job-name=bert-dp-2gpu -o name | grep launcher)
```

**What happens under the hood:**
1. The MPI Operator creates a launcher pod and 2 worker pods
2. SSH keys are auto-generated and distributed to workers
3. The launcher executes `mpirun -np 2 python train_bert.py`
4. Each worker loads BERT-Base, processes its data shard (batch size 32 per worker)
5. Gradients are synchronized via Ring-AllReduce (NCCL over TCP)
6. Metrics are printed as delimited JSON to stdout

**Expected results (3 epochs):**
- Throughput: ~85 samples/s (vs ~54 on 1 GPU)
- Scaling efficiency: 78.3%
- Validation accuracy: ~92%
- GPU utilization: ≥99.7%

### Step 4: Run Case B — GPT-2 Pipeline-Parallel Training

Case B uses PyTorch Distributed RPC to split GPT-2 Large across two GPUs via pipeline parallelism.

```bash
kubectl apply -f training/gpt2-pipeline-rpc.yaml

# Monitor both pods
kubectl logs -f gpt2-pipeline-rpc-0  # Master + Worker1 (Stage 1)
kubectl logs -f gpt2-pipeline-rpc-1  # Worker2 (Stage 2)
```

**What happens under the hood:**
1. A headless Service provides DNS names for pod discovery
2. The Indexed Job creates 2 pods with unique `JOB_COMPLETION_INDEX` values
3. Pod 0 launches the master (rank 0) and Worker1 (rank 1, Stage 1 on GPU)
4. Pod 1 launches Worker2 (rank 2, Stage 2 on GPU)
5. GPT-2 Large is partitioned: embedding + blocks 0–11 on GPU 0, blocks 12–23 + LM head on GPU 1
6. Forward pass sends activations Stage 1 → Stage 2 via RPC; backward pass reverses
7. Gradient clipping (norm 1.0) and LR warmup prevent gradient explosion

**Expected results (3 epochs, ~7.2 hours):**
- Training loss: 2.98 → 2.08
- Peak GPU memory: 15,352 / 15,360 MB (99.9%)
- GPU utilization: Stage 1 ~94%, Stage 2 ~62%
- Pipeline bubble: 50% (synchronous)

### Step 5: Extract Metrics

Training scripts output structured JSON between delimiters for reliable extraction:

```bash
# Extract epoch-level metrics
kubectl logs <pod-name> | sed -n '/===METRICS_START===/,/===METRICS_END===/p'
```

The JSON includes: throughput, GPU utilization, memory usage, step timing breakdown, communication overhead, and cost per epoch.

### Step 6: Tear Down Infrastructure

To avoid ongoing AWS charges:

```bash
# Delete training jobs first
kubectl delete mpijob --all
kubectl delete job --all

# Destroy all infrastructure
cd infra/
terraform destroy
```

## Key Architecture Decisions

### Why MPI Operator for Data Parallelism?
The Kubeflow MPI Operator automates SSH key distribution, hostfile generation, and `mpirun` invocation — eliminating the manual setup that makes MPI on Kubernetes painful. It introduces the `MPIJob` CRD that manages the full launcher/worker lifecycle.

### Why Indexed Jobs for Pipeline Parallelism?
PyTorch RPC requires each process to know its rank at startup. Kubernetes Indexed Jobs assign a unique `JOB_COMPLETION_INDEX` to each pod, which the entrypoint script maps to RPC ranks. A headless Service provides stable DNS names (`pod-0.service`, `pod-1.service`) for RPC endpoint discovery.

### Why Separate Pods (Not Multi-GPU Nodes)?
Using separate single-GPU nodes (g4dn.xlarge) instead of multi-GPU nodes (g4dn.12xlarge) tests the harder case — network-bound gradient exchange over TCP rather than NVLink. This is representative of most real cloud deployments where multi-GPU instances are expensive or unavailable.

## Stability Measures for Large Model Training

The initial GPT-2 Large training run failed at step 2,660 due to gradient explosion (loss → NaN). Three measures were implemented:

1. **Gradient clipping** at norm 1.0 via a custom `ClippedAdamW` optimizer
2. **Linear LR warmup** over 5% of total steps (445 steps)
3. **NaN detection** that halts training after 5 consecutive NaN losses

These are implemented in `train_gpt2_pipeline.py` and are recommended for any large Transformer training.

## Results Summary

### Case A: Data Parallelism Scaling (BERT-Base on SST-2)

| Metric | 1 GPU | 2 GPUs |
|---|---|---|
| Throughput (samples/s) | 54.2 | 84.9 |
| Speedup | 1.0x | 1.57x |
| Scaling Efficiency | — | 78.3% |
| Wall Time (s) | 1,244 | 793 |
| Val Accuracy | 92.0% | 92.1% |
| GPU Utilization | 99.7% | 100% |
| AllReduce Overhead | 6.8% | 16.6% |
| Cost per Epoch | $0.18 | $0.23 |

### Case B: Pipeline Parallelism (GPT-2 Large on WikiText-2)

| Metric | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---|---|---|
| Train Loss | 2.980 | 2.481 | 2.085 |
| Val Perplexity | 19.73 | 20.96 | 23.83 |
| Throughput (samp/s) | 2.7 | 2.7 | 2.7 |
| GPU Util (Stage1/Stage2) | 94/63% | 95/64% | 95/62% |
| Peak GPU Memory | 15,356 MB | 15,352 MB | 15,352 MB |
| Cost per Epoch | $2.53 | $2.58 | $2.54 |

## Choosing a Parallelism Strategy

```
Model fits in single GPU VRAM?
├─ Yes → Use Data Parallelism (Horovod, DDP)
└─ No
   └─ Model fits across K GPUs?
      ├─ Yes → Use Pipeline Parallelism (PyTorch RPC, GPipe)
      └─ No
         └─ High-bandwidth interconnect (NVLink)?
            ├─ Yes → Use 3D Parallelism (Megatron-LM, DeepSpeed)
            └─ No  → Use ZeRO / FSDP memory optimization
```

## References

- [Tutorial Paper](main.tex) — Full IEEE tutorial with theory, implementation, and analysis
- [Horovod](https://github.com/horovod/horovod) — Distributed training framework for data parallelism
- [PyTorch Distributed RPC](https://pytorch.org/docs/stable/rpc.html) — RPC-based distributed training
- [Kubeflow MPI Operator](https://github.com/kubeflow/mpi-operator) — Kubernetes operator for MPI jobs
- [Terraform AWS EKS Module](https://registry.terraform.io/modules/terraform-aws-modules/eks/aws) — EKS cluster provisioning

## License

This project was developed as a capstone for the HPCC program at KFUPM, 2026.
