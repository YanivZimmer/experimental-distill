#!/bin/bash
# Submit training job to Vertex AI

# Configuration
PROJECT_ID="your-gcp-project-id"
REGION="us-central1"
BUCKET_NAME="your-bucket-name"
JOB_NAME="distill-training-$(date +%Y%m%d-%H%M%S)"
IMAGE_URI="gcr.io/${PROJECT_ID}/distill-trainer:latest"

# GPU configuration (flexible - change as needed)
MACHINE_TYPE="g2-standard-8"  # 1x L4 GPU
# Alternatives:
# MACHINE_TYPE="a2-highgpu-1g"  # 1x A100
# MACHINE_TYPE="n1-standard-8"  # CPU only for testing

# Build and push Docker image
echo "Building Docker image..."
docker build -t ${IMAGE_URI} .
docker push ${IMAGE_URI}

# Submit to Vertex AI
echo "Submitting training job..."
gcloud ai custom-jobs create \
  --region=${REGION} \
  --display-name=${JOB_NAME} \
  --worker-pool-spec=machine-type=${MACHINE_TYPE},replica-count=1,accelerator-type=NVIDIA_L4,accelerator-count=1,container-image-uri=${IMAGE_URI} \
  --service-account=your-service-account@${PROJECT_ID}.iam.gserviceaccount.com

echo "Job submitted: ${JOB_NAME}"
echo "Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
