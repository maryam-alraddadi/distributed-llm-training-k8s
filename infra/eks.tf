module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 21.0"

  name               = "hpcc-capstone-cluster"
  kubernetes_version = "1.29" # Always stick to a recent stable version

  vpc_id                   = module.vpc.vpc_id
  subnet_ids               = module.vpc.public_subnets  # Node subnets
  control_plane_subnet_ids = module.vpc.public_subnets  # Control plane subnets (must match for communication)

  # Enable public access so you can run kubectl from your laptop
  endpoint_public_access = true

  # Allow your current identity to manage the cluster
  enable_cluster_creator_admin_permissions = true

  # Required EKS addons - vpc-cni must be installed BEFORE nodes
  addons = {
    coredns = {}
    kube-proxy = {}
    vpc-cni = {
      before_compute = true  # Critical: install before node groups
    }
    eks-pod-identity-agent = {
      before_compute = true
    }
  }

  eks_managed_node_groups = {
    # 1. System Node Group: Small CPUs for running system tools (CoreDNS, Metrics Server)
    system_nodes = {
      ami_type       = "AL2023_x86_64_STANDARD"  # Amazon Linux 2023
      instance_types = ["t3.medium"]
      min_size       = 1
      max_size       = 2
      desired_size   = 1
    }

    # 2. Training Node Group: GPU nodes - start with 1 to fit within 8 vCPU quota
    gpu_training_nodes = {
      name = "gpu-training-pool"
      
      ami_type       = "AL2_x86_64_GPU"  # GPU AMI (still AL2-based)
      instance_types = ["g4dn.xlarge"]   # 4 vCPUs, 16GB RAM, 1 T4 GPU

      min_size     = 1
      max_size     = 2  
      desired_size = 2 
      
      # Labels help Kubernetes schedule ONLY training jobs on these nodes
      labels = {
        "role" = "training-worker"
        "accelerator" = "nvidia-t4"
      }
    }
  }
}