import os
import shutil
import yaml

BASE_DIR = r"c:\Users\ASUS\Downloads\ADRI\crop_monitoring_system\dataset"
SOURCE_DIR = os.path.join(BASE_DIR, "ds1_ws8FQ5jd9r")

# Create target train/valid directories
for split in ['train', 'valid', 'test']:
    os.makedirs(os.path.join(BASE_DIR, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, split, 'labels'), exist_ok=True)

# Copy files from ds1_ws8FQ5jd9r if target is empty
for split in ['train', 'valid', 'test']:
    src_img_dir = os.path.join(SOURCE_DIR, split, 'images')
    src_lbl_dir = os.path.join(SOURCE_DIR, split, 'labels')
    tgt_img_dir = os.path.join(BASE_DIR, split, 'images')
    tgt_lbl_dir = os.path.join(BASE_DIR, split, 'labels')
    
    if os.path.exists(src_img_dir):
        for f in os.listdir(src_img_dir):
            shutil.copy2(os.path.join(src_img_dir, f), os.path.join(tgt_img_dir, f))
    if os.path.exists(src_lbl_dir):
        for f in os.listdir(src_lbl_dir):
            shutil.copy2(os.path.join(src_lbl_dir, f), os.path.join(tgt_lbl_dir, f))

# Create dataset/data.yaml
data_yaml_content = {
    'path': BASE_DIR.replace('\\', '/'),
    'train': 'train/images',
    'val': 'valid/images',
    'test': 'test/images',
    'nc': 6,
    'names': [
        'Caterpillar and Semilooper Pest Attack',
        'Healthy',
        'Rust',
        'Spectoria_Brown_Spot',
        'frog_eye',
        'mosaic'
    ]
}

yaml_path = os.path.join(BASE_DIR, "data.yaml")
with open(yaml_path, 'w') as f:
    yaml.dump(data_yaml_content, f, default_flow_style=False)

print(f"✅ YOLO dataset populated successfully in {BASE_DIR}")
print(f"✅ Created {yaml_path}")
