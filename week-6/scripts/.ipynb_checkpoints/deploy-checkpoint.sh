#!/bin/bash

#1. Set env
export PROJECT_ID=geometric-gamma-472903-q2
export REGION=us-central1            # artifact registry region
export REPO_NAME=iris-api
export GKE_CLUSTER=iris-cluster
export GKE_ZONE=us-central1-a
gcloud config set project $PROJECT_ID

#2. Enable required APIs
gcloud services enable artifactregistry.googleapis.com \
  container.googleapis.com \
  containerregistry.googleapis.com \
  iam.googleapis.com
  
#3. Create Artifact Registry (Docker)
gcloud artifacts repositories create $REPO_NAME \
  --repository-format=docker \
  --location=$REGION \
  --description="Docker repo for IRIS API"

#4. Create GKE cluster
gcloud container clusters create $GKE_CLUSTER \
  --zone $GKE_ZONE \
  --num-nodes 2 \
  --machine-type e2-medium

#5.Create a Service Account for GitHub Actions and grant roles
SA_NAME=github-actions-deployer
gcloud iam service-accounts create $SA_NAME --display-name "GH Actions deployer"

# Bind roles needed: container.admin to manage GKE, artifactregistry.admin to push images, storage.objectViewer for artifacts if needed, service account user
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/container.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"

#6. Create and download the JSON key
gcloud iam service-accounts keys create ./gcp-sa-key.json \
  --iam-account=${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com

