#!/bin/bash

# --- CONFIGURATION ---
SERVICE_NAME="pharmacoforge"
IMAGE_URL="us-central1-docker.pkg.dev/xgboost-305718/dkoes/forge:latest"
REGION="us-central1" 
GPU_TYPE="nvidia-l4"  # Options: nvidia-l4, nvidia-t4 (Check region availability)
QUOTA_METRIC="NVIDIA_L4_GPUS" # Must match the GPU_TYPE (e.g. NVIDIA_T4_GPUS)

# --- SETUP ---
echo "🚀 Starting Smart Deployment for $SERVICE_NAME..."

# 1. Get Project ID
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)


# 2. If no project is set, ask the user to pick one
if [[ -z "$PROJECT_ID" || "$PROJECT_ID" == "(unset)" ]]; then
  echo "⚠️  No project is currently selected."
  echo "📋 Here are your available projects:"
  echo "------------------------------------"
  
  # List projects nicely
  gcloud projects list --format="table(projectId, name)"
  
  echo "------------------------------------"
  echo "👉 Please copy and paste the PROJECT_ID you want to use below:"
  read -p "Project ID: " USER_INPUT_PROJECT
  
  if [ -z "$USER_INPUT_PROJECT" ]; then
    echo "❌ No Project ID provided. Exiting."
    exit 1
  fi

  # Set the project
  gcloud config set project "$USER_INPUT_PROJECT"
  PROJECT_ID=$USER_INPUT_PROJECT
fi

echo "✅ Using Project: $PROJECT_ID"

# 2. Enable APIs
echo "⚙️  Enabling necessary APIs (Cloud Run, Compute Engine, Artifact Registry)..."
gcloud services enable run.googleapis.com compute.googleapis.com artifactregistry.googleapis.com

# --- QUOTA CHECK ---
echo "🔍 Checking for $GPU_TYPE quota in $REGION..."

# Fetch the specific limit for the requested GPU metric in the region
QUOTA_LIMIT=$(gcloud compute regions describe $REGION --format="json" | jq -r ".quotas[] | select(.metric == \"$QUOTA_METRIC\") | .limit")

# Treat empty output as 0
if [[ -z "$QUOTA_LIMIT" ]]; then QUOTA_LIMIT=0; fi
# Convert to integer (removes decimals like 1.0)
QUOTA_LIMIT=${QUOTA_LIMIT%.*}

# --- DEPLOYMENT LOGIC ---
if [ "$QUOTA_LIMIT" -gt 0 ]; then
  echo "✅ GPU Quota detected ($QUOTA_LIMIT available). Deploying with GPU power! ⚡"
  
  gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_URL \
    --region $REGION \
    --allow-unauthenticated \
    --port 8080 \
    --cpu 4 \
    --memory 16Gi \
    --no-cpu-throttling \
    --gpu 1 \
    --min-instances=0 \
    --max-instances=1 \
    --no-gpu-zonal-redundancy \
    --gpu-type $GPU_TYPE

else
  echo "⚠️  NO GPU QUOTA DETECTED for $GPU_TYPE in $REGION."
  echo "   (Your account has a limit of 0 for this GPU type)."
  echo ""
  echo "   ... Falling back to CPU-only deployment."
  echo "   NOTE: The app will run slowly."
  
  # CPU-Only Fallback Command
  gcloud run deploy ${SERVICE_NAME}-nogpu \
    --image $IMAGE_URL \
    --region $REGION \
    --allow-unauthenticated \
    --port 8080 \
    --cpu 16 \
    --memory 16Gi \
    --min-instances=0 \
    --max-instances=1 

fi

echo "🎉 Deployment attempt finished."
if [ "$QUOTA_LIMIT" -eq 0 ]; then
  echo "💡 TO ENABLE GPU: Request a quota increase for '$QUOTA_METRIC' in '$REGION' here:"
  echo "   https://console.cloud.google.com/iam-admin/quotas?project=$PROJECT_ID"
fi