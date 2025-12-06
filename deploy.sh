#!/bin/bash

echo "================================"
echo "ML Pipeline Deployment"
echo "================================"

# Build Docker image
echo "Building Docker image..."
docker build -t ml-pipeline .

# Run container
echo "Starting container..."
docker run -d -p 5000:5000 --name ml-pipeline-api ml-pipeline

echo ""
echo "================================"
echo "Deployment Complete!"
echo "================================"
echo ""
echo "API is running at: http://localhost:5000"
echo ""
echo "Test it:"
echo "  curl http://localhost:5000/health"
echo ""
echo "Make prediction:"
echo "  curl -X POST http://localhost:5000/predict -H 'Content-Type: application/json' -d @sample_request.json"
echo ""
echo "Stop container:"
echo "  docker stop ml-pipeline-api"
echo "  docker rm ml-pipeline-api"
echo ""
