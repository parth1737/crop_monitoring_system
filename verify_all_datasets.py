import os
import yaml
import glob

BASE_DIR = r"c:\Users\ASUS\Downloads\ADRI\crop_monitoring_system\dataset"
DOWNLOADS_DIR = os.path.join(BASE_DIR, "downloads")

print("=== DOWNLOADED ZIP FILES ===")
for fname in os.listdir(DOWNLOADS_DIR):
    fpath = os.path.join(DOWNLOADS_DIR, fname)
    if os.path.isfile(fpath):
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"File: {fname} | Size: {size_mb:.2f} MB")

print("\n=== EXTRACTED DATASETS INSPECTION ===")
for entry in os.listdir(BASE_DIR):
    epath = os.path.join(BASE_DIR, entry)
    if os.path.isdir(epath) and entry != "downloads":
        print(f"\nFolder: {entry}")
        
        # Check YAML file
        yaml_files = glob.glob(os.path.join(epath, "**", "*.yaml"), recursive=True)
        if yaml_files:
            for yf in yaml_files:
                print(f"  Config YAML: {os.path.relpath(yf, epath)}")
                try:
                    with open(yf, 'r') as f:
                        yd = yaml.safe_load(f)
                        print(f"  - Classes ({len(yd.get('names', {}))}): {yd.get('names')}")
                        print(f"  - Train path: {yd.get('train')}")
                        print(f"  - Val path: {yd.get('val')}")
                except Exception as e:
                    print(f"  - Error reading YAML: {e}")
        else:
            print("  No YAML file found.")
            
        # Count images
        img_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_count = 0
        for root, dirs, files in os.walk(epath):
            image_count += sum(1 for f in files if f.lower().endswith(img_extensions))
        print(f"  Total Images: {image_count}")
