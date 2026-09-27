import os
import zipfile
import yaml

base_dir = r"c:\Users\ASUS\Downloads\ADRI\crop_monitoring_system\dataset"
zip_path = os.path.join(base_dir, "downloads", "dataset_3_agrisense.zip")
extract_dir = os.path.join(base_dir, "agrisense_soyabean_leaf_disease")

if os.path.exists(zip_path):
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"Zip file size: {size_mb:.2f} MB")
    
    # Try testing zip validity
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            bad_file = zf.testzip()
            if bad_file:
                print(f"Corrupted file found in zip: {bad_file}")
            else:
                print("Zip file integrity verified! Extracting...")
                zf.extractall(extract_dir)
                print(f"Successfully extracted to {extract_dir}")
                
                # Check data.yaml in extracted dir
                yaml_path = os.path.join(extract_dir, "data.yaml")
                if os.path.exists(yaml_path):
                    with open(yaml_path, 'r') as f:
                        data = yaml.safe_load(f)
                        print("Classes found in dataset 3:")
                        print(data.get('names', {}))
    except Exception as e:
        print(f"Zip test/extract status: {e}")
else:
    print("Zip file not found yet.")
