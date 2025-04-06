variable "vpc_id" {
  description = "VPC ID for the ALB"
  type        = string
}

variable "public_subnets" {
  description = "Public subnets to deploy the ALB"
  type        = list(string)
}

variable "domain_name" {
  description = "Your subdomain (e.g. app.mydomain.com)"
  type        = string
}

variable "route53_zone_id" {
  description = "The Route 53 hosted zone ID"
  type        = string
}
