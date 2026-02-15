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

# Helm and Kubernetes providers use exec auth so they work without kubeconfig
# and work in the same apply that creates the cluster (aws eks get-token runs at connect time)
provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)

    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name, "--region", "us-east-1"]
    }
  }
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)

  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name, "--region", "us-east-1"]
  }
}
