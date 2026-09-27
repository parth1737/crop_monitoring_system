import os
import subprocess
import zipfile
import yaml

BASE_DIR = r"c:\Users\ASUS\Downloads\ADRI\crop_monitoring_system\dataset"
DOWNLOADS_DIR = os.path.join(BASE_DIR, "downloads")
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

DATASETS = [
    {
        "id": "ds3_agrisense",
        "url": "https://app.roboflow.com/ds/Nafwi86t2f?key=lABmEJXXWC",
        "zip": os.path.join(DOWNLOADS_DIR, "dataset_3_agrisense.zip"),
        "extract": os.path.join(BASE_DIR, "ds3_agrisense"),
    },
    {
        "id": "ds1_ws8FQ5jd9r",
        "url": "https://app.roboflow.com/ds/ws8FQ5jd9r?key=ejXubKkLga",
        "zip": os.path.join(DOWNLOADS_DIR, "ds1_ws8FQ5jd9r.zip"),
        "extract": os.path.join(BASE_DIR, "ds1_ws8FQ5jd9r"),
    },
    {
        "id": "ds2_nrvyBIv66y",
        "url": "https://app.roboflow.com/ds/nrvyBIv66y?key=0AHGlkedxO",
        "zip": os.path.join(DOWNLOADS_DIR, "ds2_nrvyBIv66y.zip"),
        "extract": os.path.join(BASE_DIR, "ds2_nrvyBIv66y"),
    },
    {
        "id": "ds4_G8h9jAjMPR",
        "url": "https://app.roboflow.com/ds/G8h9jAjMPR?key=PHng5Iolef",
        "zip": os.path.join(DOWNLOADS_DIR, "ds4_G8h9jAjMPR.zip"),
        "extract": os.path.join(BASE_DIR, "ds4_G8h9jAjMPR"),
    },
]

def download_dataset(item):
    zip_path = item["zip"]
    url = item["url"]
    
    if os.path.exists(zip_path) and os.path.getsize(zip_path) > 100000:
        print(f"[{item['id']}] Zip file already downloaded ({os.path.getsize(zip_path) / (1024*1024):.2f} MB).")
        return
    
    print(f"[{item['id']}] Downloading from {url}...")
    cmd = ["curl.exe", "-L", "-o", zip_path, url]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(zip_path):
        print(f"[{item['id']}] Download complete! Size: {os.path.getsize(zip_path) / (1024*1024):.2f} MB.")
    else:
        print(f"[{item['id']}] Download failed. Output: {res.stderr}")

def extract_dataset(item):
    zip_path = item["zip"]
    extract_dir = item["extract"]
    
    if os.path.exists(extract_dir) and len(os.listdir(extract_dir)) > 0:
        print(f"[{item['id']}] Already extracted into {extract_dir}.")
        return
        
    if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 100000:
        print(f"[{item['id']}] Skip extract: zip missing or empty.")
        return
        
    print(f"[{item['id']}] Extracting {zip_path} to {extract_dir}...")
    os.makedirs(extract_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
        print(f"[{item['id']}] Extracted successfully!")
    except Exception as e:
        print(f"[{item['id']}] Extraction failed: {e}")

if __name__ == "__main__":
    for item in DATASETS:
        download_dataset(item)
        extract_dataset(item)
    print("\n--- ALL DOWNLOADS AND EXTRACTIONS COMPLETED ---")
