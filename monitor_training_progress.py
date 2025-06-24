#!/usr/bin/env python3
"""
Monitor training progress for constrained overfitting
Run this on remote server to track key metrics
"""
import re
import sys
from pathlib import Path
from datetime import datetime

def parse_log_line(line):
    """Extract key metrics from log line"""
    metrics = {}
    
    # Extract step
    step_match = re.search(r'Step: (\d+)/(\d+)', line)
    if step_match:
        metrics['step'] = int(step_match.group(1))
        metrics['total_steps'] = int(step_match.group(2))
    
    # Extract PnL
    pnl_match = re.search(r'PnL: \$([+-]?\d+(?:\.\d+)?)', line)
    if pnl_match:
        metrics['pnl'] = float(pnl_match.group(1))
    
    # Extract cash
    cash_match = re.search(r'Cash: \$(\d+(?:\.\d+)?)', line)
    if cash_match:
        metrics['cash'] = float(cash_match.group(1))
    
    # Extract position
    pos_match = re.search(r'Pos: (\d+(?:\.\d+)?)', line)
    if pos_match:
        metrics['position'] = float(pos_match.group(1))
    
    # Extract action
    action_match = re.search(r'Action: ([+-]?\d+\.\d+)', line)
    if action_match:
        metrics['action'] = float(action_match.group(1))
    
    # Extract violations
    violations_match = re.search(r'Violations: (\d+)', line)
    if violations_match:
        metrics['violations'] = int(violations_match.group(1))
    
    return metrics

def analyze_training_progress(log_file):
    """Analyze training progress from log file"""
    if not Path(log_file).exists():
        print(f"Log file not found: {log_file}")
        return
    
    print(f"\n{'='*80}")
    print(f"CONSTRAINED OVERFITTING PROGRESS ANALYSIS")
    print(f"Log: {log_file}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # Track metrics over time
    steps_data = []
    cash_depletion_events = []
    profitable_episodes = []
    constraint_violations = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if 'Step:' in line:
                metrics = parse_log_line(line)
                if metrics:
                    steps_data.append(metrics)
                    
                    # Track cash depletion
                    if 'cash' in metrics and metrics['cash'] < 100:
                        cash_depletion_events.append(metrics)
                    
                    # Track profitable episodes
                    if 'pnl' in metrics and metrics['pnl'] > 0:
                        profitable_episodes.append(metrics)
                    
                    # Track violations
                    if 'violations' in metrics:
                        constraint_violations.append(metrics)
    
    if not steps_data:
        print("No training data found yet.")
        return
    
    # Calculate statistics
    latest = steps_data[-1]
    total_steps = latest.get('step', 0)
    
    # Progress
    print(f"TRAINING PROGRESS:")
    print(f"  Current Step: {total_steps:,}")
    if 'total_steps' in latest:
        progress = total_steps / latest['total_steps'] * 100
        print(f"  Progress: {progress:.1f}%")
    
    # Performance
    print(f"\nPERFORMANCE:")
    if 'pnl' in latest:
        print(f"  Latest PnL: ${latest['pnl']:,.2f}")
        return_pct = latest['pnl'] / 100000 * 100
        print(f"  Return: {return_pct:.2f}%")
    
    # Cash management
    print(f"\nCASH MANAGEMENT:")
    recent_depletion = [e for e in cash_depletion_events if e['step'] > total_steps - 10000]
    print(f"  Cash depletion events (last 10k steps): {len(recent_depletion)}")
    if recent_depletion:
        avg_action_when_broke = sum(e.get('action', 0) for e in recent_depletion) / len(recent_depletion)
        buy_attempts_when_broke = sum(1 for e in recent_depletion if e.get('action', 0) > 0)
        print(f"  Buy attempts when broke: {buy_attempts_when_broke}/{len(recent_depletion)}")
        print(f"  Avg action when broke: {avg_action_when_broke:.3f}")
    
    # Constraint violations
    if constraint_violations:
        recent_violations = [v for v in constraint_violations if v['step'] > total_steps - 10000]
        if recent_violations:
            avg_violations = sum(v['violations'] for v in recent_violations) / len(recent_violations)
            print(f"\nCONSTRAINT LEARNING:")
            print(f"  Avg violations (last 10k steps): {avg_violations:.1f}")
    
    # Profitability trend
    print(f"\nPROFITABILITY TREND:")
    profit_checkpoints = [0, 50000, 100000, 150000, 200000]
    for checkpoint in profit_checkpoints:
        if checkpoint <= total_steps:
            checkpoint_data = [s for s in steps_data if abs(s['step'] - checkpoint) < 5000]
            if checkpoint_data and 'pnl' in checkpoint_data[0]:
                pnl = checkpoint_data[0]['pnl']
                print(f"  Step {checkpoint:,}: ${pnl:,.2f} ({pnl/1000:.1f}%)")
    
    # Success indicators
    print(f"\n{'='*80}")
    print("SUCCESS INDICATORS:")
    
    is_profitable = latest.get('pnl', 0) > 0
    low_violations = len(recent_depletion) < 10
    good_cash_mgmt = latest.get('cash', 0) > 1000
    
    print(f"  ✓ Profitable: {'YES' if is_profitable else 'NO'}")
    print(f"  ✓ Low violations: {'YES' if low_violations else 'NO'}")
    print(f"  ✓ Good cash management: {'YES' if good_cash_mgmt else 'NO'}")
    
    if is_profitable and low_violations and good_cash_mgmt:
        print(f"\n🎉 OVERFITTING SUCCESSFUL! Model is learning to profit on June 20th!")
    else:
        print(f"\n⏳ Still training... Keep monitoring for improvements.")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        # Default to latest log
        log_file = "logs/constrained_overfit_june20/training.log"
    
    analyze_training_progress(log_file)