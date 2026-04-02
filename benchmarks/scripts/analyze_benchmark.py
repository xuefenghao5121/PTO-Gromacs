#!/usr/bin/env python3
"""
Benchmark analysis script for PTO-Gromacs
Analyzes baseline vs PTO timings and generates report
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

def parse_timings(filename):
    """Parse timings from file"""
    timings = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith("TIME:"):
                parts = line.strip().split()
                timings.append(float(parts[1]))
    return np.array(timings)

def parse_tile_sweep(filename):
    """Parse tile size sweep results"""
    results = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith("TILE_SIZE:"):
                parts = line.strip().split()
                tile_size = int(parts[1])
                time = float(parts[2])
                results.append((tile_size, time))
    return results

def stats_summary(timings):
    """Calculate statistics"""
    mean = np.mean(timings)
    std = np.std(timings)
    min_t = np.min(timings)
    median = np.median(timings)
    return {
        'mean': mean,
        'std': std,
        'min': min_t,
        'median': median,
        'n': len(timings)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', help='Baseline timings file', required=True)
    parser.add_argument('--pto', help='PTO timings file', required=True)
    parser.add_argument('--tile-sweep', help='Tile size sweep results', required=True)
    parser.add_argument('--output', help='Output markdown report', required=True)
    args = parser.parse_args()
    
    # Parse data
    baseline_timings = parse_timings(args.baseline)
    pto_timings = parse_timings(args.pto)
    tile_sweep = parse_tile_sweep(args.tile_sweep)
    
    # Calculate statistics
    baseline_stats = stats_summary(baseline_timings)
    pto_stats = stats_summary(pto_timings)
    
    # Calculate speedup
    speedup = baseline_stats['mean'] / pto_stats['mean']
    speedup_pct = (speedup - 1) * 100
    
    # Find best tile size
    tile_sweep_arr = np.array(tile_sweep)
    best_idx = np.argmin(tile_sweep_arr[:, 1])
    best_tile = tile_sweep_arr[best_idx, 0]
    best_time = tile_sweep_arr[best_idx, 1]
    
    # Generate report
    with open(args.output, 'w') as f:
        f.write("# PTO-Gromacs Benchmark Report\n\n")
        f.write("## Summary\n\n")
        f.write(f"- Baseline mean: **{baseline_stats['mean']:.2f} ms** (±{baseline_stats['std']:.2f})\n")
        f.write(f"- PTO mean: **{pto_stats['mean']:.2f} ms** (±{pto_stats['std']:.2f})\n")
        f.write(f"- Speedup: **{speedup:.2f}x** (+{speedup_pct:.1f}%)\n\n")
        
        f.write("## Statistics\n\n")
        f.write("| Metric | Baseline | PTO Optimized |\n")
        f.write("|--------|----------|---------------|\n")
        f.write(f"| Mean (ms) | {baseline_stats['mean']:.2f} | {pto_stats['mean']:.2f} |\n")
        f.write(f"| Median (ms) | {baseline_stats['median']:.2f} | {pto_stats['median']:.2f} |\n")
        f.write(f"| Min (ms) | {baseline_stats['min']:.2f} | {pto_stats['min']:.2f} |\n")
        f.write(f"| Std Dev | {baseline_stats['std']:.2f} | {pto_stats['std']:.2f} |\n")
        f.write(f"| Iterations | {baseline_stats['n']} | {pto_stats['n']} |\n\n")
        
        f.write("## Tile Size Sweep\n\n")
        f.write("| Tile Size | Time (ms) |\n")
        f.write("|-----------|----------|\n")
        for tile_size, time in sorted(tile_sweep):
            marker = "**" if tile_size == best_tile else ""
            f.write(f"| {marker}{tile_size}{marker} | {marker}{time:.2f}{marker} |\n")
        f.write(f"\nBest tile size: **{best_tile}** with time **{best_time:.2f} ms**\n\n")
        
        f.write("## Conclusion\n\n")
        if speedup > 1:
            f.write(f"✅ **PTO optimization provides positive speedup** of {speedup:.2f}x ({speedup_pct:.1f}%)\n")
        else:
            f.write(f"⚠️  PTO optimization did not improve performance on this system/size\n")
            f.write(f"   Check: tile size selection, cache size, compiler flags\n")
    
    print(f"Report generated: {args.output}")
    print(f"Speedup: {speedup:.2f}x (+{speedup_pct:.1f}%)")
    print(f"Best tile size: {best_tile}")

if __name__ == '__main__':
    main()
