import os
import zipfile
import urllib.request

def download_file(url, target_path):
    print(f"Downloading {url} to {target_path}...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response, open(target_path, 'wb') as out_file:
        data = response.read()
        out_file.write(data)
    print(f"Downloaded {os.path.getsize(target_path)} bytes.")

def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to}...")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")

if __name__ == "__main__":
    base_dir = r"c:\Users\ASUS\Downloads\ADRI\crop_monitoring_system\dataset"
    downloads_dir = os.path.join(base_dir, "downloads")
    os.makedirs(downloads_dir, exist_ok=True)
    
    # Dataset 3
    ds3_url = "https://app.roboflow.com/ds/Nafwi86t2f?key=lABmEJXXWC"
    ds3_zip = os.path.join(downloads_dir, "dataset_3_agrisense.zip")
    ds3_extract = os.path.join(base_dir, "agrisense_soyabean_leaf_disease")
    
    download_file(ds3_url, ds3_zip)
    extract_zip(ds3_zip, ds3_extract)
