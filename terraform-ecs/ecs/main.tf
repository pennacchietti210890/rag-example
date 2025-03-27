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
      image     = "640168414029.dkr.ecr.eu-north-1.amazonaws.com/backend/rag_example:latest" # replace if needed
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
    }
  ])
}

resource "aws_ecs_service" "backend" {
  name            = "backend-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.backend.arn
  launch_type     = "FARGATE"
  desired_count   = 0

  network_configuration {
    subnets         = var.public_subnets
    security_groups = [aws_security_group.backend_sg.id]
    assign_public_ip = true
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
    cidr_blocks      = ["0.0.0.0/0"]
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
      image     = "640168414029.dkr.ecr.eu-north-1.amazonaws.com/frontend/rag_example:latest" # Replace if needed
      portMappings = [
        {
          containerPort = 8501
          protocol      = "tcp"
        }
      ]
      environment = [
        {
          name  = "BACKEND_URL"
          value = "http://13.61.177.237:8000" # replace this!
        }
      ]
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
    cidr_blocks = ["0.0.0.0/0"] # Public access for now
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
    subnets         = var.public_subnets
    security_groups = [aws_security_group.frontend_sg.id]
    assign_public_ip = true
  }
}