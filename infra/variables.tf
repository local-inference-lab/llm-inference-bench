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
}

variable "max_capacity" {
  description = "Maximum number of GPU instances"
  type        = number
  default     = 1
}

variable "gpu_count" {
  description = "Number of GPUs to reserve per ECS task"
  type        = number
  default     = 1
}

variable "container_image" {
  description = "Container image for inference server"
  type        = string
  default     = "vllm/vllm-openai:latest"
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
