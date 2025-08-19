variable "region" {
  type    = string
  default = "us-east-1"
}

variable "ssh_key_name" {
  type    = string
  default = "ai-poc-key"
}

variable "management_instance_type" {
  type    = string
  default = "t2.micro"
}

variable "s3_bucket" {
  type    = string
  default = "ai-poc-gpt-data-REPLACE_WITH_UNIQUE"
}

variable "project_name" {
  type    = string
  default = "mini-gpt"
}

variable "aws_key_name" {
  type    = string
  description = "Your AWS keypair name for EC2 SSH access"
}