import warnings
warnings.filterwarnings('ignore')

from scripts.data_model import NLPDataInput, NLPDataOutput, ImageDataInput, ImageDataOutput
from scripts import s3

from fastapi import FastAPI
import uvicorn
import os
import time

import torch
from transformers import pipeline
from transformers import AutoImageProcessor  # -> like Tokenizer

model_ckpt = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(model_ckpt, use_fast=True)

app = FastAPI()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

####### Download / Load ML Models (controlled with force_download) ##########

# Set this to True to force re-download of all models from S3 on startup.
# Keep False to use local copies if present.
force_download = False

# list of model folders expected in S3 and local ml-models/
model_names = [
    "tinybert-sentiment-analysis",
    "tinybert-disaster-tweet",
    "pose_classifer_analyzer"
]

local_base = "ml-models"

# Download any missing models (or all if force_download True)
if force_download:
    print("force_download=True -> downloading all models from S3")
    s3.download_all_models(local_base_path=local_base)
else:
    # Only download folders that are missing locally
    for m in model_names:
        local_path = os.path.join(local_base, m)
        if not os.path.isdir(local_path):
            print(f"Local model folder missing: {local_path} -> downloading from S3")
            s3.download_dir(local_path, m)
        else:
            print(f"Local model folder exists: {local_path} (skipping download)")

# Initialize pipelines from local folders
sentiment_model = pipeline('text-classification', model=os.path.join(local_base, "tinybert-sentiment-analysis"), device=device)
tweeter_model   = pipeline('text-classification', model=os.path.join(local_base, "tinybert-disaster-tweet"), device=device)
pose_model      = pipeline('image-classification', model=os.path.join(local_base, "pose_classifer_analyzer"), device=device, image_processor=image_processor)

######## Download / Load ENDS  #############


@app.get("/")
def read_root():
    return "Hello! I am up!!!"


@app.post("/api/v1/sentiment_analysis")
def sentiment_analysis(data: NLPDataInput):
    start = time.time()
    output = sentiment_model(data.text)
    end = time.time()
    prediction_time = int((end - start) * 1000)

    labels = [x['label'] for x in output]
    scores = [x['score'] for x in output]

    output = NLPDataOutput(model_name="tinybert-sentiment-analysis",
                           text=data.text,
                           labels=labels,
                           scores=scores,
                           prediction_time=prediction_time)

    return output


@app.post("/api/v1/disaster_classifier")
def disaster_classifier(data: NLPDataInput):
    start = time.time()
    output = tweeter_model(data.text)
    end = time.time()
    prediction_time = int((end - start) * 1000)

    labels = [x['label'] for x in output]
    scores = [x['score'] for x in output]

    output = NLPDataOutput(model_name="tinybert-disaster-tweet",
                           text=data.text,
                           labels=labels,
                           scores=scores,
                           prediction_time=prediction_time)

    return output


@app.post("/api/v1/pose_classifier")
def pose_classifier(data: ImageDataInput):
    start = time.time()
    urls = [str(x) for x in data.url]
    output = pose_model(urls)
    end = time.time()
    prediction_time = int((end - start) * 1000)

    # pipeline returns a list per image; take top result if present
    labels = [preds[0]['label'] if preds else None for preds in output]
    scores = [preds[0]['score'] if preds else None for preds in output]

    output = ImageDataOutput(model_name="pose_classifer_analyzer",
                             url=data.url,
                             labels=labels,
                             scores=scores,
                             prediction_time=prediction_time)

    return output


if __name__ == "__main__":
    uvicorn.run(app="app:app", port=8502, reload=True)
