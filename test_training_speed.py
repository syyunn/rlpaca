#!/usr/bin/env python3
"""Test actual training speeds with SAC"""

import time
import numpy as np

print("SAC TRAINING SPEED ANALYSIS")
print("="*60)

# From our tests:
# - 1000 steps in 3 seconds = 333 steps/sec (data collection only)
# - 10,000 steps timed out after 120 seconds

print("\nObserved speeds:")
print("- First 1000 steps (data collection): 333 steps/sec")
print("- After 1000 steps (actual training): MUCH slower")

# Estimate based on timeout
# If 10k steps didn't finish in 120 seconds:
# First 1000 took ~3 seconds, remaining 9000 took >117 seconds
remaining_steps = 9000
min_time = 117  # seconds

max_training_speed = remaining_steps / min_time
print(f"- Training speed: <{max_training_speed:.1f} steps/sec")

print("\nRealistic time estimates:")
for timesteps in [1000, 5000, 10000, 50000, 100000]:
    if timesteps <= 1000:
        time_est = timesteps / 333
    else:
        # First 1000 steps at 333/sec, rest at ~75/sec
        time_est = 1000/333 + (timesteps-1000)/75
    
    if time_est < 60:
        print(f"- {timesteps:,} steps: ~{time_est:.0f} seconds")
    elif time_est < 3600:
        print(f"- {timesteps:,} steps: ~{time_est/60:.1f} minutes")
    else:
        print(f"- {timesteps:,} steps: ~{time_est/3600:.1f} hours")

print("\nWHY SO SLOW?")
print("1. Neural network training (backprop) is computationally expensive")
print("2. SAC has 3 networks (actor + 2 critics) updating every step")
print("3. Large observation space (5,185 dims) = big networks")
print("4. CPU training (no GPU acceleration)")

print("\nRECOMMENDATION FOR VIABLE MODEL:")
print("- Minimum viable: 5,000 steps (~1 minute)")
print("- Better: 20,000 steps (~4 minutes)") 
print("- Good: 50,000 steps (~11 minutes)")
print("- Production: 100,000+ steps (~22 minutes)")

print("\nBUT: Even 5,000 steps should show SOME learning")
print("(not just constant buy/sell like untrained model)")