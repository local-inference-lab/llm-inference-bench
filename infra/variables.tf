variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type for GPU nodes"
  type        = string
  default     = "g6e.xlarge"
}

variable "desired_capacity" {
  description = "Desired number of GPU instances"
  type        = number
  default     = 1

  validation {
    condition     = var.desired_capacity >= 0 && floor(var.desired_capacity) == var.desired_capacity
    error_message = "desired_capacity must be a non-negative integer."
  }
}

variable "max_capacity" {
  description = "Maximum number of GPU instances"
  type        = number
  default     = 1

  validation {
    condition = (
      var.max_capacity >= 1
      && floor(var.max_capacity) == var.max_capacity
      && var.max_capacity >= var.desired_capacity
    )
    error_message = "max_capacity must be an integer >= 1 and >= desired_capacity."
  }
}

variable "gpu_count" {
  description = "Number of GPUs to reserve per ECS task"
  type        = number
  default     = 1

  validation {
    condition     = var.gpu_count >= 1 && floor(var.gpu_count) == var.gpu_count
    error_message = "gpu_count must be an integer >= 1."
  }
}

variable "container_image" {
  description = "Container image for inference server. Pinned to a specific tag for reproducible benchmarks."
  type        = string
  default     = "vllm/vllm-openai:v0.21.0-cu129-ubuntu2404"
}

variable "container_port" {
  description = "Port exposed by the inference container"
  type        = number
  default     = 8000
}

variable "task_cpu" {
  description = "CPU units for the ECS task"
  type        = number
  default     = 4096
}

variable "task_memory" {
  description = "Memory (MiB) for the ECS task"
  type        = number
  default     = 16384
}

variable "cluster_name" {
  description = "Name of the ECS cluster"
  type        = string
  default     = "gpu-inference"
}

variable "allowed_ingress_cidrs" {
  description = "CIDR blocks allowed to reach the inference port. Defaults to the VPC CIDR (internal-only). Override to expose externally (e.g. workstation IPs)."
  type        = list(string)
  default     = ["10.0.0.0/16"]

  validation {
    condition     = length(var.allowed_ingress_cidrs) > 0
    error_message = "allowed_ingress_cidrs must contain at least one CIDR block."
  }
}
