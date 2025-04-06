resource "aws_ecs_cluster" "main" {
  name = "rag-app-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Name = "rag-app-cluster"
  }
}

resource "aws_cloudwatch_log_group" "backend" {
  name              = "/ecs/backend"
  retention_in_days = 7
}

resource "aws_cloudwatch_log_group" "frontend" {
  name              = "/ecs/frontend"
  retention_in_days = 7
}

resource "aws_ecs_task_definition" "backend" {
  family                   = "backend-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"
  memory                   = "1024"
  execution_role_arn       = "arn:aws:iam::640168414029:role/ecsTaskExecutionRole"


  container_definitions = jsonencode([
    {
      name      = "backend"
      image     = "640168414029.dkr.ecr.eu-north-1.amazonaws.com/backend/rag_example:v2" # replace if needed
      portMappings = [
        {
          containerPort = 8000
          protocol      = "tcp"
        }
      ]
      environment = [
        {
          name  = "ENV"
          value = "production"
        }
      ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = "/ecs/backend"
        awslogs-region        = "eu-north-1"
        awslogs-stream-prefix = "ecs"
      }
    }
    }
  ])
}

resource "aws_service_discovery_private_dns_namespace" "main" {
  name        = "rag-app.local"
  description = "Private namespace for service discovery"
  vpc         = var.vpc_id
}

resource "aws_service_discovery_service" "backend" {
  name = "backend-service"

  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.main.id
    dns_records {
      type = "A"
      ttl  = 10
    }
    routing_policy = "MULTIVALUE"
  }

  health_check_custom_config {
    failure_threshold = 1
  }
}

resource "aws_ecs_service" "backend" {
  name            = "backend-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.backend.arn
  launch_type     = "FARGATE"
  desired_count   = 0

  network_configuration {
    subnets         = [var.private_subnets[0]]
    security_groups = [aws_security_group.backend_sg.id]
    assign_public_ip = false
  }

  service_registries {
    registry_arn = aws_service_discovery_service.backend.arn
  }
}

resource "aws_security_group" "backend_sg" {
  name        = "backend-sg"
  description = "Allow HTTP traffic on port 8000"
  vpc_id      = var.vpc_id

  ingress {
    description      = "Allow from anywhere (dev only)"
    from_port        = 8000
    to_port          = 8000
    protocol         = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port        = 0
    to_port          = 0
    protocol         = "-1"
    cidr_blocks      = ["0.0.0.0/0"]
  }
}

resource "aws_ecs_task_definition" "frontend" {
  family                   = "frontend-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"
  memory                   = "1024"
  execution_role_arn       = "arn:aws:iam::640168414029:role/ecsTaskExecutionRole" # same as backend

  container_definitions = jsonencode([
    {
      name      = "frontend"
      image     = "640168414029.dkr.ecr.eu-north-1.amazonaws.com/frontend/rag_example:v2" # Replace if needed
      portMappings = [
        {
          containerPort = 8501
          protocol      = "tcp"
        }
      ]
      environment = [
        {
          name  = "BACKEND_URL"
          value = "http://backend-service.rag-app.local:8000" # replace this!
        }
      ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = "/ecs/frontend"
        awslogs-region        = "eu-north-1"
        awslogs-stream-prefix = "ecs"
      }
    }
    }
  ])
}

resource "aws_security_group" "frontend_sg" {
  name        = "frontend-sg"
  description = "Allow access to Streamlit port 8501"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 8501
    to_port     = 8501
    protocol    = "tcp"
    security_groups  = [var.alb_sg_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

}

resource "aws_ecs_service" "frontend" {
  name            = "frontend-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.frontend.arn
  launch_type     = "FARGATE"
  desired_count   = 0

  network_configuration {
    subnets         = [var.private_subnets[0]]
    security_groups = [aws_security_group.frontend_sg.id]
    assign_public_ip = false  # Because ALB handles exposure
  }

  load_balancer {
    target_group_arn = var.frontend_tg_arn  # or aws_lb_target_group.frontend_tg.arn
    container_name   = "frontend"
    container_port   = 8501
  }

  depends_on = [
    var.alb_listener_dependency
  ]

  lifecycle {
  prevent_destroy = true
  }
}