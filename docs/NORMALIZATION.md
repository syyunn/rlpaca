# Normalization Strategy

## Overview

Proper input normalization is critical for neural network training. Without it, large input values can saturate activation functions, preventing learning.

## The Problem We Solved

Initially, our RL agent was stuck outputting constant actions (always -1.0 or +1.0). Investigation revealed that raw Unix timestamps (e.g., 1,750,426,240) were causing network saturation:

```
tanh(weight × 1.75e9) → exactly ±1.0 
→ zero gradient → no learning
```

## Our Normalization Approach

### 1. Timestamps
- **Raw**: Unix epoch (1.75+ billion seconds)
- **Normalized**: Hours since market open (0-6.5 range)
- **Code**: `(timestamp - market_open).total_seconds() / 3600`

### 2. Volume
- **Raw**: Millions of shares per minute bar
- **Normalized**: Divided by 1e6
- **Rationale**: NVDA typically trades 50-200M shares/day

### 3. Sizes (Bid/Ask)
- **Raw**: Shares (100-10,000)
- **Normalized**: Divided by 1,000
- **Rationale**: Typical quote sizes are in thousands

### 4. Trade Counts
- **Raw**: 100-5,000 trades per minute
- **Normalized**: Divided by 1,000
- **Rationale**: Brings to single/double digits

### 5. Position
- **Raw**: Number of shares
- **Normalized**: Divided by 100
- **Rationale**: For $100k account at ~$140/share ≈ 700 shares max

### 6. Capital/Portfolio Value
- **Raw**: Dollar amounts
- **Normalized**: As fraction of initial capital
- **Rationale**: Makes it scale-invariant

## Implementation

All normalization constants are defined in `src/rl/realistic_offline_env.py`:

```python
VOLUME_NORMALIZER = 1e6      # Convert to millions
SIZE_NORMALIZER = 1e3        # Convert to thousands  
TRADE_COUNT_NORMALIZER = 1e3 # Convert to thousands
POSITION_NORMALIZER = 100    # Convert to hundreds
```

## Results

After normalization:
- Observation values: ~0-200 range (instead of billions)
- Actions: Dynamic range (-0.5 to 0.8) instead of saturated ±1.0
- Training: Successful learning with standard networks

## Key Takeaway

Always check your input ranges! Even theoretically learnable transformations (like scaling) can prevent practical learning due to gradient saturation.