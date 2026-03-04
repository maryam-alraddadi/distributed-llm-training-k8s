# Base: official Horovod image — PyTorch, CUDA, OpenMPI, NCCL, and Horovod
# are all pre-compiled. Much smaller and avoids source-build issues.
FROM horovod/horovod:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Upgrade pip so it pulls pre-built wheels instead of compiling from source
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# HuggingFace libraries for BERT model + dataset loading
RUN pip install --no-cache-dir \
    transformers==4.36.2 \
    datasets==2.16.1 \
    accelerate==0.25.0 \
    scikit-learn==1.3.2 \
    pynvml

WORKDIR /workspace
COPY train_bert.py .
