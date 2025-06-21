#!/usr/bin/env python3
"""Test the 5k step model to see if it learned anything"""

import numpy as np
from stable_baselines3 import SAC

# Load the model
model = SAC.load('../rlpaca/fast_model_5000.zip')
print("Loaded 5,000 step model")

# Create random observations to test different scenarios
n_tests = 20
observations = np.random.randn(n_tests, 5185).astype(np.float32)

# Modify key features to create different scenarios
for i in range(n_tests):
    # Vary position state (last 5 features)
    observations[i, -5] = i / n_tests  # Position from 0 to 1
    observations[i, -4] = 1 - (i / n_tests)  # Cash from 1 to 0
    observations[i, -2] = np.random.uniform(-0.05, 0.05)  # Recent return

print("\nTesting model on different scenarios:")
print("Position | Cash | Action | Interpretation")
print("-" * 50)

actions = []
for i, obs in enumerate(observations):
    action, _ = model.predict(obs, deterministic=True)
    position = obs[-5]
    cash = obs[-4]
    actions.append(action[0])
    
    if abs(action[0]) < 0.1:
        interpretation = "HOLD"
    elif action[0] > 0:
        interpretation = "BUY"
    else:
        interpretation = "SELL"
    
    print(f"{position:.2f}     | {cash:.2f} | [{action[0]:6.3f}, {action[1]:5.1f}] | {interpretation}")

# Check if model learned anything
action_std = np.std(actions)
action_mean = np.mean(actions)

print(f"\nStatistics:")
print(f"Mean action: {action_mean:.3f}")
print(f"Std action: {action_std:.3f}")
print(f"Min action: {min(actions):.3f}")
print(f"Max action: {max(actions):.3f}")

if action_std < 0.01:
    print("\n❌ VERDICT: Model outputs nearly constant actions")
    print("   It hasn't learned any meaningful trading strategy")
    print("   This is basically a 'always sell' bot")
else:
    print("\n✅ VERDICT: Model shows some variation in actions")
    print("   There's hope it learned something!")

print("\n💡 INSIGHT: After only 5k steps, the model hasn't learned much.")
print("   It's like a trader who only worked for 13 days (5000/390).")
print("   For comparison, GPT-3.5 trained on billions of examples!")