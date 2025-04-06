output "alb_dns_name" {
  value = aws_lb.frontend_alb.dns_name
}

output "frontend_tg_arn" {
  value = aws_lb_target_group.frontend_tg.arn
}

output "listener_arn" {
  value = aws_lb_listener.frontend_http.arn
}

output "alb_sg_id" {
  value = aws_security_group.alb_sg.id
}