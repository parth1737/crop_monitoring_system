"""
Centralized Configuration for Soybean Crop Monitoring and Pesticide Recommendation System.
Configured for Edge-AI real-time inference on drone hardware & laptop simulation.
"""

import os
import torch

# Directory Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
RESULTS_DIR = os.path.join(OUTPUTS_DIR, "results")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Ensure required output directories exist
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# Model Settings
MODEL_PATH = os.path.join(OUTPUTS_DIR, "best.pt")
PRETRAINED_MODEL = "yolov8n.pt"  # Nano model selected for low-latency edge deployment
DATASET_YAML = os.path.join(DATASET_DIR, "data.yaml")

# Knowledge Base Path
CSV_PATH = os.path.join(BASE_DIR, "pesticide_knowledge_base.csv")

# Real-Time Processing Settings
CAMERA_SOURCE = 0  # 0 for webcam, or string path to video file (e.g., "flight_video.mp4")
CONFIDENCE_THRESHOLD = 0.5
IMAGE_SIZE = 640
EPOCHS = 50
BATCH_SIZE = 16

# Device Configuration with Automatic Fallback Logic
if torch.cuda.is_available():
    DEVICE = "0"
    DEVICE_NAME = torch.cuda.get_device_name(0)
else:
    DEVICE = "cpu"
    DEVICE_NAME = "CPU"

# Visualization Settings
import cv2
VNDVI_COLORMAP = cv2.COLORMAP_JET
CROP_TYPE = "Soybean"

# VNDVI Health Thresholds (Visible NDVI from RGB)
VNDVI_HEALTHY_THRESHOLD = 0.3
VNDVI_MODERATE_THRESHOLD = 0.1

print(f"[CONFIG LOADED] Target Crop: {CROP_TYPE} | Compute Device: {DEVICE_NAME} ({DEVICE})")
