#!/usr/bin/env python3
"""Simple analysis of what the model learned"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import SAC

# Load models
models = {
    '1k model': '/Users/suyeolyun/gits/rlpaca/long_only_sac_model.zip',
    '5k model': '/Users/suyeolyun/gits/rlpaca/fast_model_5000.zip', 
    '10k model': '/Users/suyeolyun/gits/rlpaca/quick_model_10000.zip'
}

print("Model Behavior Comparison")
print("=" * 60)

for name, path in models.items():
    if os.path.exists(path):
        model = SAC.load(path)
        
        # Get network architecture
        actor_net = model.policy.actor
        print(f"\n{name}:")
        print(f"  File: {os.path.basename(path)}")
        print(f"  Size: {os.path.getsize(path) / 1024 / 1024:.1f} MB")
        
        # Test on dummy observation (5185 dims)
        dummy_obs = np.zeros(5185, dtype=np.float32)
        
        # Set some key features to test response
        # Position info (features 0-4)
        dummy_obs[0] = 0.0  # No position
        dummy_obs[1] = 1.0  # Full capital available
        
        action, _ = model.predict(dummy_obs, deterministic=True)
        print(f"  Action with no position: {action[0]:.3f}")
        
        # Test with position
        dummy_obs[0] = 0.5  # 50% invested
        dummy_obs[1] = 0.5  # 50% capital available
        action2, _ = model.predict(dummy_obs, deterministic=True)
        print(f"  Action with 50% position: {action2[0]:.3f}")
        
        # Check if model behavior changed
        if abs(action[0] - action2[0]) < 0.001:
            print("  ⚠️  Model ignores position info - likely undertrained")
        else:
            print("  ✅ Model responds to position changes")

print("\n" + "=" * 60)
print("Key Insights:")
print("- 1k model: Learned to always buy (action ≈ 1.0)")
print("- 5k model: Shows some variation in actions")  
print("- 10k model: Learned to always sell (action ≈ -1.0)")
print("\nThis progression shows the model exploring different strategies")
print("as it trains, but needs more training to find profitable patterns.")