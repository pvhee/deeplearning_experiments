#!/bin/bash

BUCKET='svhn-digits-ml'
PROJECT='svhn-digits'
REGION='us-central1'
MODEL_NAME='svhn_digits'
MODEL_VERSION='v2'
EXPORT_VERSION='1521490959'
EXPORT_DIR='export/'$EXPORT_VERSION
RUNTIME_VERSION='1.5'

# Set default project and region
gcloud config set project $PROJECT
gcloud config set compute/region $REGION

# Create bucket
if ! gsutil ls | grep -q gs://${BUCKET}/; then
  gsutil mb -l ${REGION} gs://${BUCKET}
fi

# Upload model to bucket
# Use -n (no-clobber) to not overwrite existing files
gsutil cp -n -a public-read -R ${EXPORT_DIR} gs://${BUCKET}

# Clean out bucket if you want to upload a new version
#gsutil rm gs://${BUCKET}/**

# Fetch the model location
MODEL_LOCATION=$(gsutil ls gs://${BUCKET}/ | tail -1)
echo "Using model location: ${MODEL_LOCATION}"

# Create model in Cloud ML
#gcloud ml-engine models create ${MODEL_NAME} --regions $REGION
gcloud ml-engine versions create ${MODEL_VERSION} --model ${MODEL_NAME} --origin ${MODEL_LOCATION} --runtime-version ${RUNTIME_VERSION}

# Get info about this model & version
gcloud ml-engine versions describe ${MODEL_VERSION} --model ${MODEL_NAME}

# Make predictions using the prediction API
#JSON_INPUT='data.json'
#gcloud ml-engine predict --model-dir ${MODEL_NAME} --version ${MODEL_VERSION} --json-instances ${JSON_INPUT}
