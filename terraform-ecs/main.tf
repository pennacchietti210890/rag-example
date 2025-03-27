provider "aws" {
  region = "eu-north-1"  # Change to your region
}

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  required_version = ">= 1.3.0"
}

# Include modules
module "networking" {
  source = "./networking"
}

module "ecs" {
  source = "./ecs"
  vpc_id = module.networking.vpc_id
  public_subnets = module.networking.public_subnets
}
