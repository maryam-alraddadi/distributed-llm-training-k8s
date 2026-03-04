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
  # Allow all TCP traffic between nodes in the cluster (required for MPI
  # communication: SSH from launcher→workers and orted back-connections).
  # Without this, the system and GPU node groups have isolated security groups.
  node_security_group_additional_rules = {
    ingress_self_all = {
      description = "Node-to-node all traffic"
      protocol    = "-1"
      from_port   = 0
      to_port     = 0
      type        = "ingress"
      self        = true
    }
  }

  eks_managed_node_groups = {
    # 1. System Node Group
    system_nodes = {
      ami_type       = "AL2023_x86_64_STANDARD"
      instance_types = ["t3.medium"]
      min_size       = 1
      max_size       = 2
      desired_size   = 1

      # Increase root volume to 100GB
      block_device_mappings = {
        xvda = {
          device_name = "/dev/xvda"
          ebs = {
            volume_size = 100
            volume_type = "gp3" # gp3 is faster and cheaper than older gp2
          }
        }
      }
    }

    # 2. Training Node Group
    gpu_training_nodes = {
      name = "gpu-training-pool"
      ami_type       = "AL2_x86_64_GPU"
      instance_types = ["g4dn.xlarge"]
      min_size       = 0
      max_size       = 2
      desired_size   = 2 
      
      labels = {
        "role" = "training-worker"
        "accelerator" = "nvidia-t4"
      }

      #  Increase root volume to 100GB
      block_device_mappings = {
        xvda = {
          device_name = "/dev/xvda"
          ebs = {
            volume_size = 100
            volume_type = "gp3"
          }
        }
      }
    }
  }
}