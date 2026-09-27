"""
app.py - Single-Command Launch Web Server for Soybean Disease AI System
======================================================================
Usage:
    python app.py
Loads YOLOv8 model, starts web server on http://127.0.0.1:5000,
and opens the 5-image disease detection interface in your web browser.
"""

import os
import sys
import glob
import uuid
import time
import cv2
import traceback
import numpy as np
import webbrowser
from threading import Timer
from flask import Flask, render_template, request, jsonify, send_from_directory
from ultralytics import YOLO

from recommendation_engine import PesticideRecommendationEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "best.pt")
STATIC_OUTPUT_DIR = os.path.join(BASE_DIR, "static", "outputs")
os.makedirs(STATIC_OUTPUT_DIR, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")

print("="*70)
print("🌾 INITIALIZING SOYBEAN DISEASE DETECTION WEB SERVER...")
print("="*70)

if not os.path.exists(MODEL_PATH):
    print(f"❌ ERROR: Model weights file not found at '{MODEL_PATH}'!")
    sys.exit(1)

print(f"📦 Loading pre-trained YOLOv8 model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
rec_engine = PesticideRecommendationEngine()
print("✅ AI Model & Recommendation Engine loaded successfully!\n")

def sanitize_json(obj):
    """Recursively converts NumPy data types into standard Python types."""
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8, np.integer)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_json(obj.tolist())
    elif isinstance(obj, dict):
        return {str(k): sanitize_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_json(i) for i in obj]
    return obj

def process_single_image(img_bgr, slot_id):
    # Ensure 3-channel BGR format
    if len(img_bgr.shape) == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    elif img_bgr.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

    h, w, _ = img_bgr.shape

    # 1. YOLOv8 Disease Detection (conf=0.15 to capture all disease spots including Frogeye Leaf Spot)
    results = model.predict(img_bgr, conf=0.15, verbose=False)
    boxes = results[0].boxes
    
    detected_diseases = []
    
    # Generate high-quality bounding box visualization using Ultralytics plot()
    annotated_img = results[0].plot(line_width=2, font_size=1.0)
    
    if len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0].item()) if hasattr(box, 'cls') else 0
            cls_name = str(results[0].names[cls_id])
            conf = float(box.conf[0].item())
            detected_diseases.append({
                "disease": cls_name,
                "confidence": round(conf * 100, 1)
            })

    # 2. Extract Primary Disease & Recommendation
    primary_disease = str(detected_diseases[0]["disease"]) if detected_diseases else "Healthy"
    recommendation = rec_engine.get_recommendation(primary_disease)

    # 3. Create 2-Panel Output Image (Panel 1: YOLOv8 Bounding Boxes | Panel 2: Disease Prescription Report)
    card_h = 540
    card_w = 420
    card_panel = np.zeros((card_h, card_w, 3), dtype=np.uint8) + 25
    
    cv2.putText(card_panel, "SOYBEAN DISEASE REPORT", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.line(card_panel, (15, 45), (405, 45), (100, 100, 100), 1)
    
    # Detected Diseases Section
    cv2.putText(card_panel, "DETECTED LEAF DISEASES:", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 2)
    y_pos = 100
    if detected_diseases:
        unique_diseases = {}
        for d in detected_diseases:
            d_name = d["disease"]
            d_conf = d["confidence"]
            if d_name not in unique_diseases or d_conf > unique_diseases[d_name]:
                unique_diseases[d_name] = d_conf
                
        for d_name, d_conf in unique_diseases.items():
            d_line = f"• {d_name} ({d_conf}%)"
            if len(d_line) > 36:
                d_line = d_line[:34] + ".."
            cv2.putText(card_panel, d_line, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            y_pos += 22
    else:
        cv2.putText(card_panel, "• Healthy Crop Leaf", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
        y_pos += 22

    # Treatment Prescription Section
    cv2.line(card_panel, (15, y_pos + 5), (405, y_pos + 5), (100, 100, 100), 1)
    y_pos += 30
    cv2.putText(card_panel, "RECOMMENDED TREATMENT:", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2)
    y_pos += 25
    
    rec_lines = [
        f"Product: {recommendation.get('recommended_pesticide')}",
        f"Active: {recommendation.get('active_ingredient')}",
        f"Dosage: {recommendation.get('dosage_per_ha')} / ha",
        f"Water Vol: {recommendation.get('water_volume_per_ha')} / ha",
        f"Interval: {recommendation.get('spray_interval_days')} days",
        f"Safety Buffer: {recommendation.get('safety_buffer_m')} meters"
    ]
    for line in rec_lines:
        cv2.putText(card_panel, str(line), (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
        y_pos += 22
        
    cv2.line(card_panel, (15, y_pos + 5), (405, y_pos + 5), (100, 100, 100), 1)
    y_pos += 25
    cv2.putText(card_panel, "Application Notes:", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
    y_pos += 20
    note = str(recommendation.get('application_notes', ''))
    words = note.split()
    curr_line = ""
    for w_word in words:
        if len(curr_line + " " + w_word) < 36:
            curr_line += " " + w_word
        else:
            cv2.putText(card_panel, curr_line.strip(), (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
            y_pos += 18
            curr_line = w_word
    if curr_line and y_pos < card_h - 10:
        cv2.putText(card_panel, curr_line.strip(), (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    # 4. Resize YOLOv8 Bounding Box Image to 640x540 and combine 2 panels ONLY
    p1 = cv2.resize(annotated_img, (640, card_h))
    cv2.putText(p1, "[1] YOLOv8 Disease Bounding Boxes", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    combined_2_panel = np.hstack([p1, card_panel])

    timestamp = int(time.time() * 1000)
    out_filename = f"bbox_slot_{slot_id}_{uuid.uuid4().hex[:6]}.jpg"
    out_path = os.path.join(STATIC_OUTPUT_DIR, out_filename)
    cv2.imwrite(out_path, combined_2_panel)
    
    return sanitize_json({
        "slot_id": int(slot_id),
        "detected_diseases": detected_diseases,
        "disease_count": len(detected_diseases),
        "primary_disease": primary_disease,
        "recommendation": recommendation,
        "output_image_url": f"/static/outputs/{out_filename}?t={timestamp}"
    })

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/get_demo_images")
def get_demo_images():
    val_dir = os.path.join(BASE_DIR, "dataset", "valid", "images")
    img_files = glob.glob(os.path.join(val_dir, "*.jpg")) + glob.glob(os.path.join(val_dir, "*.png"))
    demo_urls = []
    timestamp = int(time.time() * 1000)
    if img_files:
        for idx, img_p in enumerate(img_files[:5]):
            fname = f"demo_{idx+1}_{os.path.basename(img_p)}"
            dst = os.path.join(STATIC_OUTPUT_DIR, fname)
            if not os.path.exists(dst):
                img = cv2.imread(img_p)
                if img is not None:
                    cv2.imwrite(dst, img)
            demo_urls.append(f"/static/outputs/{fname}?t={timestamp}")
    return jsonify({"status": "success", "images": demo_urls})

@app.route("/api/predict_slots", methods=["POST"])
def predict_slots():
    try:
        results_list = []
        val_dir = os.path.join(BASE_DIR, "dataset", "valid", "images")
        demo_imgs = glob.glob(os.path.join(val_dir, "*.jpg")) + glob.glob(os.path.join(val_dir, "*.png"))

        uploaded_found = False
        for i in range(1, 6):
            file_key = f"slot_{i}"
            if file_key in request.files and request.files[file_key].filename != '':
                uploaded_found = True
                file_obj = request.files[file_key]
                in_bytes = np.frombuffer(file_obj.read(), np.uint8)
                img_bgr = cv2.imdecode(in_bytes, cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    res = process_single_image(img_bgr, i)
                    results_list.append(res)

        # Fallback to demo images if no user uploads provided
        if not uploaded_found and demo_imgs:
            for i in range(1, min(6, len(demo_imgs) + 1)):
                img_p = demo_imgs[i-1]
                img_bgr = cv2.imread(img_p)
                if img_bgr is not None:
                    res = process_single_image(img_bgr, i)
                    results_list.append(res)

        clean_results = sanitize_json(results_list)
        return jsonify({"status": "success", "results": clean_results})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.errorhandler(500)
def server_error(e):
    return jsonify({"status": "error", "message": f"Server error: {str(e)}"}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({"status": "error", "message": "Endpoint not found"}), 404

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == "__main__":
    print("🚀 Starting Web Server on http://127.0.0.1:5000")
    print("🌐 Opening web interface in your default browser...")
    Timer(1.5, open_browser).start()
    app.run(host="127.0.0.1", port=5000, debug=False)
