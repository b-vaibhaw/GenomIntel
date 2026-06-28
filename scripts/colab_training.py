#!/usr/bin/env python3
"""
colab_training.py — Automated Google Colab Execution Script.
This script automates mounting Google Drive, extracting the project, installing 
dependencies, downloading the full 2.5GB ClinVar dataset, training the 
GenomicAttentionClassifier on a T4 GPU, and saving the weights.

How to use in Colab:
1. Upload your project zip to Google Drive.
2. In a Colab notebook cell, paste:
   !python colab_training.py
"""

import os
import sys
import shutil
import subprocess

def run_cmd(cmd):
    """Run a shell command and print outputs in real-time."""
    print(f"\n[RUNNING] {cmd}")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        print(f"❌ Error: Command failed with exit code {process.returncode}")
        return False
    return True

def main():
    print("=============================================================")
    # 1. Mount Google Drive
    print("Step 1: Mounting Google Drive...")
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✔ Google Drive mounted successfully.")
    except ImportError:
        print("❌ Error: Not running inside Google Colab environment. Mount aborted.")
        sys.exit(1)

    # 2. Locate and Unzip Project
    print("\nStep 2: Extracting Project Archive...")
    drive_path = "/content/drive/MyDrive/try_new.zip"
    extract_path = "/content/biointel"
    
    if not os.path.exists(drive_path):
        print(f"❌ Error: Could not find 'try_new.zip' at the root of your Google Drive.")
        print("Please upload your project zip archive to your Google Drive root directory and try again.")
        sys.exit(1)
        
    if os.path.exists(extract_path):
        print(f"Clearing existing extraction folder: {extract_path}")
        shutil.rmtree(extract_path)
        
    os.makedirs(extract_path, exist_ok=True)
    if not run_cmd(f"unzip -q {drive_path} -d {extract_path}"):
        sys.exit(1)
    print("✔ Project archive extracted successfully.")

    # Change working directory to project root
    os.chdir(extract_path)
    print(f"Working directory set to: {os.getcwd()}")

    # 3. Install Package Dependencies
    print("\nStep 3: Installing Python dependencies...")
    if not run_cmd("pip install streamlit pandas plotly scikit-learn statsmodels patsy cryptography python-dotenv PyYAML torch"):
        sys.exit(1)
    print("✔ Dependencies installed successfully.")

    # 4. Initialize SQLite Schema
    print("\nStep 4: Initializing SQLite database schema...")
    if not run_cmd("python db/init_sqlite.py"):
        sys.exit(1)
    print("✔ Database schema initialized.")

    # 5. Download and Parse Full NCBI ClinVar Dataset
    print("\nStep 5: Downloading and streaming the entire 2.5GB NCBI ClinVar dataset...")
    # --limit -1 indicates no limits: parses all 3.5 million variants.
    if not run_cmd("python scripts/scale_dataset.py --limit -1"):
        sys.exit(1)
    print("✔ Full database scaled and variants table seeded.")

    # 6. Train the Custom Genomic Attention Model on GPU
    print("\nStep 6: Training GenomicAttentionClassifier model on T4 GPU...")
    # This automatically detects the GPU and trains for 30 epochs
    if not run_cmd("python scripts/genomic_model.py"):
        sys.exit(1)
    print("✔ Model training complete.")

    # 7. Backup Weights to Google Drive
    print("\nStep 7: Copying trained weights back to your Google Drive...")
    src_weights = "models/genomic_attention.pt"
    dest_weights = "/content/drive/MyDrive/genomic_attention.pt"
    
    if os.path.exists(src_weights):
        try:
            shutil.copy(src_weights, dest_weights)
            print(f"✔ Success! Weights copied to your Google Drive root: {dest_weights}")
            print("=============================================================")
            print("🎉 ENTERPRISE MODEL TRAINING SUCCESSFUL!")
            print("You can now download 'genomic_attention.pt' from your Google Drive")
            print("and place it in your local project's models/ directory.")
            print("=============================================================")
        except Exception as e:
            print(f"❌ Error copying weights file to Google Drive: {e}")
            sys.exit(1)
    else:
        print("❌ Error: Trained weights file not found. Training may have failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
