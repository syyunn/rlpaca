#!/usr/bin/env python3
"""Monitor training progress and alert when model is saved"""

import os
import time
import sys

def monitor_model(model_name, check_interval=10):
    """Monitor for model file creation"""
    model_path = f"/Users/suyeolyun/gits/rlpaca/{model_name}.zip"
    
    print(f"🔍 Monitoring for {model_name}...")
    print(f"Checking every {check_interval} seconds")
    
    start_time = time.time()
    while True:
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            elapsed = time.time() - start_time
            print(f"\n✅ Model saved! {model_name}.zip ({file_size:.1f} MB)")
            print(f"Training took: {elapsed/60:.1f} minutes")
            return True
            
        # Show progress dot
        print(".", end="", flush=True)
        time.sleep(check_interval)

if __name__ == "__main__":
    # Monitor 10k model
    monitor_model("long_only_10k", check_interval=30)