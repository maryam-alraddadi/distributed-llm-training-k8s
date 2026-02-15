# --- 1. NVIDIA Device Plugin (Crucial for GPUs) ---
# Runs only on GPU nodes to avoid CrashLoopBackOff on CPU-only nodes
resource "helm_release" "nvidia_device_plugin" {
  name       = "nvidia-device-plugin"
  repository = "https://nvidia.github.io/k8s-device-plugin"
  chart      = "nvidia-device-plugin"
  version    = "0.14.5" # Pin the version for stability
  namespace  = "kube-system"

  set {
    name  = "nodeSelector.accelerator"
    value = "nvidia-t4"
    type  = "string"
  }

  depends_on = [module.eks]
}

# --- 2. MPI Operator (Crucial for Distributed Training) ---
# Since MPI Operator is just a YAML file, we use a "null_resource" to apply it.
# This is a common "trick" in Terraform when a formal Helm chart isn't preferred.

resource "null_resource" "mpi_operator" {
  # This triggers the command whenever the cluster ID changes (i.e., new cluster)
  triggers = {
    cluster_endpoint = module.eks.cluster_endpoint
  }

  provisioner "local-exec" {
    # We use a single command to:
    # 1. Update your local kubectl to talk to the new cluster
    # 2. Apply the MPI Operator YAML
    command = <<EOT
      aws eks update-kubeconfig --region us-east-1 --name ${module.eks.cluster_name}
      kubectl apply -f https://raw.githubusercontent.com/kubeflow/mpi-operator/v0.4.0/deploy/v2beta1/mpi-operator.yaml
    EOT
  }

  depends_on = [module.eks]
}