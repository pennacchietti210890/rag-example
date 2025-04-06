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
  aws_region  = var.aws_region
}

module "ecs" {
  source = "./ecs"
  vpc_id = module.networking.vpc_id
  public_subnets = module.networking.public_subnets
  private_subnets = module.networking.private_subnets_ids
  frontend_tg_arn = module.alb.frontend_tg_arn
  alb_listener_dependency = module.alb.listener_arn  # or module.alb.frontend_listener if you expose it
  alb_sg_id = module.alb.alb_sg_id  # 👈 pass the SG ID
}

module "alb" {
  source           = "./alb"
  vpc_id           = module.networking.vpc_id
  public_subnets   = module.networking.public_subnets
  route53_zone_id  = "Z07924525SGBL8YYZAV4" 
  domain_name      = "app.rag-playground.com"
}