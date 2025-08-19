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
