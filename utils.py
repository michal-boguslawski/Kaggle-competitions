import os
import json
import subprocess
import zipfile


def download_data(competition_name: str):
    with open(".secrets", "r") as file:
        data = json.load(file)
        
    os.environ['KAGGLE_USERNAME'] = data["username"]
    os.environ['KAGGLE_KEY'] = data["key"]
    
    subprocess.run(['kaggle', 'competitions', 'download', '-c', competition_name])
    
    with zipfile.ZipFile(f"{competition_name}.zip", "r") as zip_ref:
        zip_ref.extractall(competition_name)
        
def submit_data(competition_name: str, folder: str, message: str = "Test"):
    with open(".secrets", "r") as file:
        data = json.load(file)
        
    os.environ['KAGGLE_USERNAME'] = data["username"]
    os.environ['KAGGLE_KEY'] = data["key"]
    
    file_path = os.path.join(folder, "submission.csv")
    
    subprocess.run(['kaggle', 'competitions', 'submit', '-c', competition_name, '-f', file_path, '-m', message])
    
    
if __name__ == "__main__":
    download_data('playground-series-s5e8')
