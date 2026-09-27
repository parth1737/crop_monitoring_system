"""
train_model.py - YOLOv8-nano Soybean Health & Disease Monitoring System
========================================================================

Hyperparameter Rationale (for Viva & Project Defense):
------------------------------------------------------
1. Model Architecture (yolov8n.pt):
   - YOLOv8 Nano is selected specifically for Edge-AI deployment (Jetson Nano, Raspberry Pi, Drones).
   - Contains ~3.2M parameters, requiring minimal memory while maintaining high inference speed (>30 FPS).
2. Image Resolution (imgsz = 640):
   - Standard YOLO input resolution. Retains fine spatial details of leaf spots (Frogeye, Rust, Mosaic)
   - Balances feature resolution and computational efficiency.
3. Batch Size (batch = 16):
   - Optimal batch size for GPU memory efficiency and gradient stability.
   - Includes automatic OOM retry logic (reduces to 8 or 4 dynamically if VRAM is exceeded).
4. Training Epochs (epochs = 50):
   - Sufficient for transfer learning fine-tuning on a pre-trained COCO backbone.
5. Early Stopping (patience = 10):
   - Prevents overfitting by halting training if validation loss does not improve for 10 consecutive epochs.
"""

import os
import sys
import shutil
import time
import torch
from ultralytics import YOLO

def check_dataset(yaml_path="dataset/data.yaml"):
    """Validates dataset YAML existence and required directories before training."""
    if not os.path.exists(yaml_path):
        print(f"❌ ERROR: Dataset configuration file not found at '{yaml_path}'!")
        print("💡 Suggestion: Ensure dataset is organized under 'dataset/' with 'data.yaml', 'train/', and 'valid/'.")
        sys.exit(1)
        
    print(f"✅ Dataset configuration verified at: {os.path.abspath(yaml_path)}")

def get_optimal_device():
    """Detects available hardware and selects CUDA GPU or CPU with warning."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🚀 GPU Detected: {gpu_name} (Using device='0')")
        return "0"
    else:
        print("⚠️ WARNING: No CUDA-compatible GPU detected! Falling back to CPU.")
        print("💡 Training on CPU will take longer (~1-3 hours). Consider using NVIDIA GPU or Google Colab for faster speed.")
        return "cpu"

def estimate_training_time(device, num_images, epochs):
    """Provides estimated training time based on hardware and dataset size."""
    if device == "0":
        est_seconds = (num_images / 100.0) * 1.5 * epochs
    else:
        est_seconds = (num_images / 100.0) * 12.0 * epochs
        
    minutes = int(est_seconds // 60)
    hours = round(minutes / 60.0, 1)
    if hours >= 1.0:
        return f"~{hours} hours"
    else:
        return f"~{max(1, minutes)} minutes"

def run_training(dataset_yaml="dataset/data.yaml", epochs=50, imgsz=640, initial_batch=16, patience=10):
    """Executes YOLOv8-nano model training with robust OOM fallback handling."""
    
    check_dataset(dataset_yaml)
    device = get_optimal_device()
    
    # Estimate dataset size
    train_img_dir = "dataset/train/images"
    num_train_imgs = len(os.listdir(train_img_dir)) if os.path.exists(train_img_dir) else 966
    est_time = estimate_training_time(device, num_train_imgs, epochs)
    
    print("\n" + "="*70)
    print("🌾 SOYBEAN DISEASE DETECTION - YOLOV8-NANO TRAINING INITIALIZATION")
    print("="*70)
    print(f"• Dataset Config  : {dataset_yaml}")
    print(f"• Training Images : {num_train_imgs} samples")
    print(f"• Target Epochs   : {epochs}")
    print(f"• Input Size      : {imgsz}x{imgsz}")
    print(f"• Initial Batch   : {initial_batch}")
    print(f"• Target Device   : {'NVIDIA GPU' if device == '0' else 'CPU'}")
    print(f"• Est. Time       : {est_time}")
    print("="*70 + "\n")

    # Load pre-trained YOLOv8 Nano weights
    model_name = "yolov8n.pt"
    print(f"📦 Loading pre-trained backbone: {model_name}...")
    model = YOLO(model_name)
    
    batch_size = initial_batch
    batch_options = [initial_batch, 8, 4, 2]
    
    for current_batch in batch_options:
        try:
            print(f"▶️ Attempting training with batch size = {current_batch}...")
            start_time = time.time()
            
            results = model.train(
                data=dataset_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=current_batch,
                device=device,
                patience=patience,
                project="outputs",
                name="soybean_training",
                save=True,
                exist_ok=True,
                verbose=True
            )
            
            elapsed_min = (time.time() - start_time) / 60.0
            print(f"\n🎉 Training completed successfully in {elapsed_min:.2f} minutes!")
            
            # Post-training artifact handling
            post_training_cleanup(model, results)
            return results
            
        except RuntimeError as e:
            err_msg = str(e)
            if "out of memory" in err_msg.lower() or "CUDA out of memory" in err_msg:
                print(f"⚠️ Out of Memory (OOM) error detected with batch size = {current_batch}.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if current_batch > 2:
                    print("🔄 Reducing batch size and retrying...")
                    continue
                else:
                    print("❌ OOM error persists even at minimum batch size (2).")
                    raise e
            else:
                print(f"❌ Unexpected training error: {e}")
                raise e

def post_training_cleanup(model, results):
    """Copies best weights to outputs/best.pt and prints performance metrics."""
    os.makedirs("outputs", exist_ok=True)
    best_weights_path = os.path.join("outputs", "soybean_training", "weights", "best.pt")
    target_weights_path = os.path.join("outputs", "best.pt")
    
    if os.path.exists(best_weights_path):
        shutil.copy2(best_weights_path, target_weights_path)
        print(f"✅ Best weights exported to: {os.path.abspath(target_weights_path)}")
    else:
        print(f"⚠️ Warning: {best_weights_path} not found.")

    print("\n" + "="*70)
    print("📊 MODEL TRAINING EVALUATION METRICS SUMMARY")
    print("="*70)
    
    try:
        # Extract metrics from validation run or training results
        val_results = model.val(data="dataset/data.yaml")
        map50 = val_results.results_dict.get('metrics/mAP50(B)', 0.0)
        map50_95 = val_results.results_dict.get('metrics/mAP50-95(B)', 0.0)
        precision = val_results.results_dict.get('metrics/precision(B)', 0.0)
        recall = val_results.results_dict.get('metrics/recall(B)', 0.0)
        
        print(f"• Mean Average Precision (mAP@50)    : {map50:.4f}")
        print(f"• Mean Average Precision (mAP@50-95) : {map50_95:.4f}")
        print(f"• Precision                          : {precision:.4f}")
        print(f"• Recall                             : {recall:.4f}")
    except Exception as ex:
        print(f"ℹ️ Training completed. Metrics saved in outputs/soybean_training/results.csv")
        
    print("="*70)
    print("📁 Output plots saved to: outputs/soybean_training/")
    print("   - Confusion Matrix : outputs/soybean_training/confusion_matrix.png")
    print("   - Training Loss    : outputs/soybean_training/results.png")
    print("   - PR Curve         : outputs/soybean_training/PR_curve.png")
    print("="*70 + "\n")

if __name__ == "__main__":
    run_training()
