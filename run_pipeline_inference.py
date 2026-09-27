"""
run_pipeline_inference.py
==========================
End-to-End Soybean Disease Detection & Pesticide Recommendation Pipeline.
Processes input crop imagery, detects diseases (YOLOv8-nano), queries pesticide
recommendation engine, and exports 2-panel visualization cards.
"""

import os
import glob
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from recommendation_engine import PesticideRecommendationEngine

def process_test_pipeline(num_images=5):
    base_dir = r"c:\Users\ASUS\Downloads\ADRI\crop_monitoring_system"
    model_path = os.path.join(base_dir, "outputs", "best.pt")
    valid_img_dir = os.path.join(base_dir, "dataset", "valid", "images")
    output_pred_dir = os.path.join(base_dir, "outputs", "predictions")
    os.makedirs(output_pred_dir, exist_ok=True)

    print("="*70)
    print("🌾 RUNNING SOYBEAN DISEASE DETECTION PIPELINE (2-PANEL LAYOUT)")
    print("="*70)

    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return
        
    print(f"📦 Loading trained YOLOv8 model: {model_path}")
    model = YOLO(model_path)
    rec_engine = PesticideRecommendationEngine()

    img_files = glob.glob(os.path.join(valid_img_dir, "*.jpg")) + glob.glob(os.path.join(valid_img_dir, "*.png"))
    if not img_files:
        print("❌ No test images found in validation directory.")
        return
        
    img_files = img_files[:num_images]
    print(f"📸 Found {len(img_files)} test images. Processing...\n")

    for idx, img_path in enumerate(img_files):
        filename = os.path.basename(img_path)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        # 1. YOLOv8 Disease Detection
        results = model.predict(img_path, conf=0.20, verbose=False)
        boxes = results[0].boxes
        
        detected_diseases = []
        annotated_img = results[0].plot(line_width=2, font_size=1.0)
        
        if len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0].item()) if hasattr(box, 'cls') else 0
                cls_name = str(results[0].names[cls_id])
                conf = float(box.conf[0].item())
                detected_diseases.append({"disease": cls_name, "confidence": round(conf * 100, 1)})

        primary_disease = detected_diseases[0]["disease"] if detected_diseases else "Healthy"
        recommendation = rec_engine.get_recommendation(primary_disease)

        # 2. Create 2-Panel Report Panel (Dark Theme)
        card_h = 540
        card_w = 420
        card_panel = np.zeros((card_h, card_w, 3), dtype=np.uint8) + 25
        
        cv2.putText(card_panel, "SOYBEAN HEALTH REPORT", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.line(card_panel, (15, 45), (405, 45), (100, 100, 100), 1)
        
        if detected_diseases:
            cv2.putText(card_panel, f"Status: INFECTED ({len(detected_diseases)} spots)", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            cv2.putText(card_panel, "Status: HEALTHY CANOPY", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(card_panel, "DETECTED DISEASES:", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 2)
        y_pos = 135
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
            cv2.putText(card_panel, "• No Disease Spots Detected", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
            y_pos += 22

        cv2.line(card_panel, (15, y_pos + 5), (405, y_pos + 5), (100, 100, 100), 1)
        y_pos += 30
        cv2.putText(card_panel, "PESTICIDE PRESCRIPTION:", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2)
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

        p1 = cv2.resize(annotated_img, (640, card_h))
        cv2.putText(p1, "[1] YOLOv8 Disease Bounding Boxes", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        combined_2_panel = np.hstack([p1, card_panel])

        out_path = os.path.join(output_pred_dir, f"pipeline_output_{idx+1}_{filename}")
        cv2.imwrite(out_path, combined_2_panel)
        
        print(f"[{idx+1}/{len(img_files)}] Processed: {filename}")
        print(f"   • Detected Diseases : {[d['disease'] for d in detected_diseases] if detected_diseases else 'Healthy'}")
        print(f"   • Recommendation    : {recommendation.get('recommended_pesticide')}")
        print(f"   • Saved 2-Panel Output : {out_path}\n")

    print(f"🎉 2-Panel Pipeline Execution Finished! Saved to: {output_pred_dir}\n")

if __name__ == "__main__":
    process_test_pipeline()
