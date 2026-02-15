provider "aws" {
  region = "us-east-1"
  
  # This automatically tags ALL resources with your project info
  # Great for cost tracking later!
  default_tags {
    tags = {
      Project     = "HPCC-Capstone"
      Owner       = "Maryam"
      Environment = "Development"
      ManagedBy   = "Terraform"
    }
  }
}

terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.12"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.25"
    }
  }

  # This tells Terraform: "Don't save state on my laptop. Save it in the cloud."
  backend "s3" {
    bucket         = "hpcc-capstone-tf-state-maryam"
    key            = "dev/terraform.tfstate" # The path inside the bucket
    region         = "us-east-1"
    
    # Locking prevents corruption
    dynamodb_table = "hpcc-capstone-tf-locks"
    encrypt        = true
  }
}

# 1. Get the authentication details for your new cluster
data "aws_eks_cluster" "cluster" {
  name = module.eks.cluster_name
  depends_on = [module.eks]
}

data "aws_eks_cluster_auth" "cluster" {
  name = module.eks.cluster_name
  depends_on = [module.eks]
}

# 2. Configure the Helm Provider (The "App Store" for K8s)
provider "helm" {
  kubernetes {
    host                   = data.aws_eks_cluster.cluster.endpoint
    cluster_ca_certificate = base64decode(data.aws_eks_cluster.cluster.certificate_authority[0].data)
    token                  = data.aws_eks_cluster_auth.cluster.token
  }
}

# 3. Configure the Kubernetes Provider (For raw manifests)
provider "kubernetes" {
  host                   = data.aws_eks_cluster.cluster.endpoint
  cluster_ca_certificate = base64decode(data.aws_eks_cluster.cluster.certificate_authority[0].data)
  token                  = data.aws_eks_cluster_auth.cluster.token
}
