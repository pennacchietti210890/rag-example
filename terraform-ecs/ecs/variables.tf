variable "vpc_id" {
  type = string
}

variable "public_subnets" {
  type = list(string)
}

variable "private_subnets" {
  description = "List of private subnet IDs"
  type        = list(string)
}

variable "frontend_tg_arn" {
  description = "Target group ARN for ALB"
  type        = string
}

variable "alb_listener_dependency" {
  description = "Resource to use as dependency for ALB listener"
  type        = any
}

variable "alb_sg_id" {
  description = "Security Group ID for the ALB"
  type        = string
}