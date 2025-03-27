output "ecs_cluster_id" {
  value = aws_ecs_cluster.main.id
}

output "backend_service_name" {
  value = aws_ecs_service.backend.name
}

output "frontend_service_name" {
  value = aws_ecs_service.frontend.name
}