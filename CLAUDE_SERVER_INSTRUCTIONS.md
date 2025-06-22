# Instructions for Claude Server - Action Memory Training

## Context
We've added a critical feature to help the RL model understand when its actions fail (e.g., trying to buy with no cash). The model now tracks:
- Every action attempted throughout the trading day
- Whether each action was successfully executed (1.0) or failed (0.0)

## Problem We're Solving
Previously, when the model ran out of cash, it would keep trying to buy (action=1.0) without understanding that these actions were failing. From the model's perspective, "choosing not to trade" and "trying but failing to trade" looked identical.

## What Changed
Modified `src/rl/realistic_offline_env.py`:
1. Added `action_history` list that tracks `(action, executed_flag)` for entire day
2. Expanded observation space from 5185 to 14545 dimensions (added 4680 steps × 2 features)
3. Model can now see its entire trading history and learn from failed attempts

## Training Commands

### 1. Train Balanced Entropy Model with Memory
```bash
cd /Users/suyeolyun/gits/rlpaca
python src/rl/train.py --config configs/balanced_entropy.json --date 2025-06-20 --name NVDA_balanced_with_memory
```

### 2. Train High Entropy Model with Memory
```bash
cd /Users/suyeolyun/gits/rlpaca
python src/rl/train.py --config configs/high_entropy.json --date 2025-06-20 --name NVDA_high_entropy_with_memory
```

## What to Monitor
1. **Action Diversity**: Check if actions become more conservative when cash is low
2. **Cash Management**: Model should avoid depleting cash too early
3. **Failed Action Rate**: Track how often actions fail to execute
4. **Trading Pattern**: Look for "save cash for the dip" behavior

## Expected Improvements
- Model should learn to pace its buying to avoid cash depletion
- Should recognize when actions fail and adjust strategy
- May learn to sell positions to free up cash for better opportunities
- Professional trader behavior: remember all trades and use them to inform decisions

## Testing After Training
```bash
# Test the balanced entropy model
python test_trained_model.py --model models/balanced_with_memory/NVDA_balanced_with_memory/final_model.zip

# Test the high entropy model  
python test_trained_model.py --model models/high_entropy_with_memory/NVDA_high_entropy_with_memory/final_model.zip
```

## Key Metrics to Report Back
1. Does the model still get stuck with 0 cash?
2. How many failed trades vs successful trades?
3. Does it learn to buy the dip after the initial drop?
4. Final returns and alpha generation

Good luck with the training! The key insight is that professional traders always remember their trades - now our model does too.