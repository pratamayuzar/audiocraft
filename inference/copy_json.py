import os
import shutil

def copy_json_files(src_folder, dest_folder):
    """Copies all JSON files from src_folder to dest_folder."""
    os.makedirs(dest_folder, exist_ok=True)
    
    for file_name in os.listdir(src_folder):
        if file_name.endswith(".json"):
            src_path = os.path.join(src_folder, file_name)
            dest_path = os.path.join(dest_folder, file_name)
            
            shutil.copy2(src_path, dest_path)
            print(f"✅ Copied: {file_name} -> {dest_folder}")

# Define source and destination folders
src_folder = "dataset/lagoe_44"
dest_folder = "dataset/lagoe"

# Run copy process
copy_json_files(src_folder, dest_folder)