import yaml
import os

print("=== SOYBEAN DATASET PRE-FLIGHT VERIFICATION ===")

# 1. Check data.yaml exists
yaml_path = 'dataset/data.yaml'
if not os.path.exists(yaml_path):
    print('❌ ERROR: dataset/data.yaml not found!')
    exit(1)

# 2. Read and display classes
with open(yaml_path) as f:
    data = yaml.safe_load(f)

print('✅ dataset/data.yaml exists!')
print('✅ Classes found:', data.get('names', 'NONE'))
print('✅ Number of classes:', len(data.get('names', [])))

# 3. Check folder structure
for split in ['train', 'valid']:
    img_path = f'dataset/{split}/images'
    lbl_path = f'dataset/{split}/labels'
    
    if not os.path.exists(img_path):
        print(f'❌ Missing: {img_path}')
    else:
        imgs = len([f for f in os.listdir(img_path) if f.endswith(('.jpg','.png','.jpeg'))])
        print(f'✅ {split}/images: {imgs} files')
    
    if not os.path.exists(lbl_path):
        print(f'❌ Missing: {lbl_path}')
    else:
        lbls = len([f for f in os.listdir(lbl_path) if f.endswith('.txt')])
        print(f'✅ {split}/labels: {lbls} files')

# 4. Check a sample label file
if os.path.exists('dataset/train/labels'):
    label_files = os.listdir('dataset/train/labels')
    if label_files:
        sample = os.path.join('dataset/train/labels', label_files[0])
        with open(sample) as f:
            content = f.read().strip()
        print(f'✅ Sample label file ({label_files[0]}):')
        print(f'   Format sample: {content[:100]}...')
