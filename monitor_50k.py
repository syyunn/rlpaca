#!/usr/bin/env python3
"""Monitor 50k training progress"""

import time
import os
import subprocess

start_time = time.time()
model_path = "/Users/suyeolyun/gits/rlpaca/quick_model_50000.zip"
log_path = "/Users/suyeolyun/gits/rlpaca/training_quick_50k.log"

print("🔍 Monitoring 50k training...")
print("Expected duration: ~22 minutes")
print("-" * 50)

last_size = 0
while True:
    # Check if model saved
    if os.path.exists(model_path):
        elapsed = time.time() - start_time
        size_mb = os.path.getsize(model_path) / 1024 / 1024
        print(f"\n✅ Training complete in {elapsed/60:.1f} minutes!")
        print(f"Model saved: {size_mb:.1f} MB")
        break
    
    # Check log progress
    try:
        result = subprocess.run(['tail', '-20', log_path], 
                               capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        
        # Look for progress lines
        for line in lines:
            if 'Progress:' in line and 'steps/s' in line:
                print(f"\r{line}", end='', flush=True)
                last_size = len(line)
    except:
        pass
    
    time.sleep(10)  # Check every 10 seconds
    
    # Show we're still monitoring
    elapsed = time.time() - start_time
    if int(elapsed) % 60 == 0:
        print(f"\n⏱️  Elapsed: {elapsed/60:.0f} minutes...", end='', flush=True)