import os
import requests
import tarfile

URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5vTzHBmQUaRNJQe5szCyKw/images-dataSAT.tar"
FILENAME = "images-dataSAT.tar"
DATA_FOLDER = "data/raw/" 
EXTRACTION_PATH = DATA_FOLDER 

def download_and_extract_data(url: str, filename: str, path: str):
    print(f"Bắt đầu tải xuống file: {filename}...")

    # --- Create directory if it doesn't exist ---
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Đã tạo thư mục: {path}")

    # --- Define temporary file path ---
    temp_filepath = os.path.join(path, filename)

    # --- Download and extract ---
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() 

        with open(temp_filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ Downloaded successfully: {temp_filepath}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error while downloading dataset: {e}")
        return

    # --- Extract the .tar file ---
    try:
        print(f"Start unzipping: {os.path.abspath(path)}")
        with tarfile.open(temp_filepath, "r") as tar:
            tar.extractall(path=path)
        
        print("✅ Extracted into data successfully!")
        os.remove(temp_filepath)
        print(f"Deleted temporary file: {filename}")

    except Exception as e:
        print(f"❌ Error while unzipping: {e}")



# download_and_extract_data(URL, FILENAME, EXTRACTION_PATH)
import yaml
def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Error: Configuration file does not exist: '{config_path}'")
        
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            return config
    except yaml.YAMLError as e:
        print(f"Wrong syntax in YAML file: {e}")
        return {}
    except Exception as e:
        print(f"Error while loading from config yaml: {e}")
        return {}