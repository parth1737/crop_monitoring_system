"""
ndvi_processor.py
=================
Computes Visible Normalized Difference Vegetation Index (VNDVI) from standard RGB drone imagery.
VNDVI Formula: (2 * Green - Red - Blue) / (2 * Green + Red + Blue + epsilon)
Generates pseudo-color heatmaps and quantitative crop health scores.
"""

import cv2
import numpy as np

def compute_vndvi(image_bgr):
    """
    Computes VNDVI matrix from an input BGR image.
    Values range from -1.0 to +1.0 (Higher values indicate healthier green vegetation).
    """
    img = image_bgr.astype(np.float32) / 255.0
    blue = img[:, :, 0]
    green = img[:, :, 1]
    red = img[:, :, 2]
    
    numerator = 2.0 * green - red - blue
    denominator = 2.0 * green + red + blue + 1e-6
    
    vndvi = numerator / denominator
    vndvi = np.clip(vndvi, -1.0, 1.0)
    return vndvi

def generate_vndvi_heatmap(vndvi_matrix):
    """
    Converts VNDVI matrix (-1 to +1) into an 8-bit Jet colormap visualization heatmap.
    """
    norm_vndvi = ((vndvi_matrix + 1.0) / 2.0 * 255.0).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm_vndvi, cv2.COLORMAP_JET)
    return heatmap

def get_crop_health_status(vndvi_matrix):
    """
    Evaluates mean VNDVI score and returns quantitative crop health metric & category.
    """
    mean_score = float(np.mean(vndvi_matrix))
    if mean_score >= 0.25:
        category = "HIGH VIGOR (HEALTHY CANOPY)"
        color = (0, 255, 0)
    elif mean_score >= 0.10:
        category = "MODERATE STRESS (MONITOR CLOSELY)"
        color = (0, 255, 255)
    else:
        category = "SEVERE CANOPY STRESS / DISEASE IMPACTED"
        color = (0, 0, 255)
        
    return {
        "mean_score": round(mean_score, 4),
        "status_category": category,
        "color_bgr": color
    }
