# Terraform tests using plan-mode + mocked AWS provider.
# Runs without AWS credentials; does not create real resources.

mock_provider "aws" {
  mock_data "aws_ssm_parameter" {
    defaults = {
      value = "ami-0123456789abcdef0"
    }
  }

  mock_data "aws_availability_zones" {
    defaults = {
      names = ["us-east-1a", "us-east-1b"]
    }
  }
}

run "defaults_phase1_g6e" {
  command = plan

  assert {
    condition     = aws_launch_template.gpu.instance_type == "g6e.xlarge"
    error_message = "Phase 1 default instance type should be g6e.xlarge."
  }

  assert {
    condition     = aws_launch_template.gpu.metadata_options[0].http_tokens == "required"
    error_message = "IMDSv2 must be enforced (http_tokens = \"required\")."
  }

  assert {
    condition     = aws_autoscaling_group.gpu.desired_capacity == 1
    error_message = "Default desired_capacity should be 1."
  }

  assert {
    condition = (
      contains(one(aws_security_group.gpu_nodes.ingress).cidr_blocks, "10.0.0.0/16")
      && length(one(aws_security_group.gpu_nodes.ingress).cidr_blocks) == 1
    )
    error_message = "Default ingress should be VPC-internal only (10.0.0.0/16)."
  }

  assert {
    condition     = one(aws_security_group.gpu_nodes.ingress).from_port == 8000
    error_message = "Ingress port should default to 8000."
  }

  assert {
    condition = strcontains(
      aws_ecs_task_definition.inference.container_definitions,
      "vllm/vllm-openai:v0.21.0-cu129-ubuntu2404"
    )
    error_message = "container_image must be pinned to v0.21.0-cu129-ubuntu2404 by default."
  }

  assert {
    condition = strcontains(
      aws_ecs_task_definition.inference.container_definitions,
      "\"type\":\"GPU\""
    )
    error_message = "Task definition must declare a GPU resource requirement."
  }
}

run "custom_instance_and_gpu_count" {
  command = plan

  variables {
    instance_type = "g6e.2xlarge"
    gpu_count     = 2
  }

  assert {
    condition     = aws_launch_template.gpu.instance_type == "g6e.2xlarge"
    error_message = "Custom instance_type should propagate to launch template."
  }

  assert {
    condition = strcontains(
      aws_ecs_task_definition.inference.container_definitions,
      "\"value\":\"2\""
    )
    error_message = "Custom gpu_count should appear in task definition."
  }
}

run "custom_ingress_cidrs" {
  command = plan

  variables {
    allowed_ingress_cidrs = ["203.0.113.10/32"]
  }

  assert {
    condition = (
      contains(one(aws_security_group.gpu_nodes.ingress).cidr_blocks, "203.0.113.10/32")
      && length(one(aws_security_group.gpu_nodes.ingress).cidr_blocks) == 1
    )
    error_message = "Custom allowed_ingress_cidrs should override the default."
  }
}
