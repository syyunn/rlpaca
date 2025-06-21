#!/usr/bin/env python3
"""Test trained model behavior to see if it's viable"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import SAC
import pandas as pd

# Load the model
model = SAC.load('long_only_sac_model.zip')
print("Model loaded successfully")

# Create different market scenarios
scenarios = [
    {
        "name": "Market Dip (Good Buy Opportunity)",
        "price_change": -0.02,  # 2% drop
        "recent_prices": [140, 138, 136, 134, 132],  # Downtrend
        "position": 0,  # No position
        "cash": 100000
    },
    {
        "name": "Market Rally (Good Sell Opportunity)",
        "price_change": 0.03,  # 3% gain
        "recent_prices": [130, 132, 135, 138, 140],  # Uptrend
        "position": 500,  # Have position
        "cash": 30000
    },
    {
        "name": "Sideways Market",
        "price_change": 0.0,
        "recent_prices": [135, 134, 135, 136, 135],  # Flat
        "position": 100,
        "cash": 86500
    },
    {
        "name": "Volatile Market",
        "price_change": 0.01,
        "recent_prices": [130, 140, 132, 138, 135],  # Choppy
        "position": 0,
        "cash": 100000
    }
]

print("\n" + "="*60)
print("TESTING MODEL BEHAVIOR IN DIFFERENT SCENARIOS")
print("="*60)

for scenario in scenarios:
    print(f"\n📊 Scenario: {scenario['name']}")
    print(f"   Recent prices: {scenario['recent_prices']}")
    print(f"   Current position: {scenario['position']} shares")
    print(f"   Available cash: ${scenario['cash']:,}")
    
    # Create a simplified observation (normally 5185 dims)
    # We'll create a dummy observation and focus on key features
    obs = np.random.randn(5185).astype(np.float32) * 0.1  # Small random values
    
    # Set some key features that might influence decisions
    # Last few prices in the observation
    for i, price in enumerate(scenario['recent_prices'][-5:]):
        obs[i*5] = price / 100.0  # Normalized price
    
    # Position state (last 5 features)
    obs[-5] = scenario['position'] / 1000.0  # Normalized position
    obs[-4] = scenario['cash'] / 100000.0  # Normalized cash
    obs[-3] = (scenario['position'] * scenario['recent_prices'][-1]) / 100000.0  # Market value
    obs[-2] = scenario['price_change']  # Recent return
    obs[-1] = 0.5  # Mid-day progress
    
    # Get model prediction
    action, _states = model.predict(obs, deterministic=True)
    position_delta = action[0]
    limit_offset = action[1]
    
    print(f"\n   🤖 Model Decision:")
    print(f"   Position Delta: {position_delta:.3f}")
    print(f"   Limit Offset: {limit_offset:.1f} bps")
    
    # Interpret the action
    if abs(position_delta) < 0.1:
        print(f"   → HOLD (no significant action)")
    elif position_delta > 0:
        if scenario['position'] == 0:
            print(f"   → BUY SIGNAL (strength: {abs(position_delta):.1%})")
        else:
            print(f"   → ADD TO POSITION (strength: {abs(position_delta):.1%})")
        
        # Calculate potential order size
        max_shares = scenario['cash'] / scenario['recent_prices'][-1]
        order_size = abs(position_delta) * max_shares
        print(f"   → Would buy ~{order_size:.1f} shares (${order_size * scenario['recent_prices'][-1]:,.0f})")
    else:
        if scenario['position'] > 0:
            sell_qty = abs(position_delta) * scenario['position']
            print(f"   → SELL SIGNAL (strength: {abs(position_delta):.1%})")
            print(f"   → Would sell ~{sell_qty:.1f} shares (${sell_qty * scenario['recent_prices'][-1]:,.0f})")
        else:
            print(f"   → SELL SIGNAL but no position to sell")
    
    # Assess if decision makes sense
    if scenario['name'] == "Market Dip (Good Buy Opportunity)" and position_delta > 0.5:
        print(f"   ✅ Good! Buying on dip")
    elif scenario['name'] == "Market Rally (Good Sell Opportunity)" and position_delta < -0.3:
        print(f"   ✅ Good! Taking profits on rally")
    elif scenario['name'] == "Sideways Market" and abs(position_delta) < 0.3:
        print(f"   ✅ Good! Minimal action in flat market")

print("\n" + "="*60)
print("OVERALL ASSESSMENT")
print("="*60)

# Test extreme scenarios
extreme_obs = np.zeros(5185, dtype=np.float32)
extreme_obs[-5] = 0  # No position
extreme_obs[-4] = 1  # Full cash
action_no_pos, _ = model.predict(extreme_obs, deterministic=True)

extreme_obs[-5] = 1  # Max position
extreme_obs[-4] = 0  # No cash
action_max_pos, _ = model.predict(extreme_obs, deterministic=True)

print(f"\nExtreme scenarios:")
print(f"No position, full cash → Action: {action_no_pos[0]:.3f}")
print(f"Max position, no cash → Action: {action_max_pos[0]:.3f}")

if action_no_pos[0] > 0 and action_max_pos[0] < 0:
    print("\n✅ Model shows basic sensible behavior:")
    print("   - Wants to buy when has cash and no position")
    print("   - Wants to sell when fully invested")
    print("\n🎯 VERDICT: Model is VIABLE for deployment with careful monitoring")
else:
    print("\n❌ Model behavior seems random or nonsensical")
    print("   - May need more training or better reward shaping")
    print("\n⚠️  VERDICT: Model needs more work before deployment")