#!/usr/bin/env python3
"""
对比两个训练方法的计算开销
用法: python compare_training_time.py <gakl_stats.json> <copy_stats.json>
"""

import json
import sys
import os

def load_stats(stats_file):
    """加载训练时间统计文件"""
    if not os.path.exists(stats_file):
        print(f"错误: 文件不存在: {stats_file}")
        return None
    
    with open(stats_file, 'r') as f:
        return json.load(f)

def compare_stats(gakl_stats, copy_stats):
    """对比两个方法的统计信息"""
    print("\n" + "="*80)
    print("训练时间对比分析")
    print("="*80)
    
    # 基本信息对比
    print("\n【基本信息】")
    print(f"GAKL方法 - 总步数: {gakl_stats['total_steps']}, 方法: {gakl_stats['method']}")
    print(f"COPY方法 - 总步数: {copy_stats['total_steps']}, 方法: {copy_stats['method']}")
    
    # 总时间对比
    print("\n【总训练时间对比】")
    gakl_total = gakl_stats['total_training_time_seconds']
    copy_total = copy_stats['total_training_time_seconds']
    time_diff = copy_total - gakl_total
    time_ratio = copy_total / gakl_total if gakl_total > 0 else 0
    
    print(f"GAKL方法总时间: {gakl_total:.2f}秒 ({gakl_total/60:.2f}分钟, {gakl_total/3600:.2f}小时)")
    print(f"COPY方法总时间: {copy_total:.2f}秒 ({copy_total/60:.2f}分钟, {copy_total/3600:.2f}小时)")
    print(f"时间差异: {time_diff:.2f}秒 ({time_diff/60:.2f}分钟)")
    print(f"时间比例: COPY是GAKL的 {time_ratio:.2f}倍")
    
    if time_diff > 0:
        print(f"  → COPY方法比GAKL方法慢 {time_diff:.2f}秒 ({time_diff/gakl_total*100:.1f}%)")
    else:
        print(f"  → COPY方法比GAKL方法快 {abs(time_diff):.2f}秒 ({abs(time_diff)/gakl_total*100:.1f}%)")
    
    # 每步平均时间对比
    print("\n【每步平均时间对比】")
    gakl_avg = gakl_stats['avg_step_time_seconds']
    copy_avg = copy_stats['avg_step_time_seconds']
    avg_diff = copy_avg - gakl_avg
    avg_ratio = copy_avg / gakl_avg if gakl_avg > 0 else 0
    
    print(f"GAKL方法平均每步: {gakl_avg:.4f}秒")
    print(f"COPY方法平均每步: {copy_avg:.4f}秒")
    print(f"每步时间差异: {avg_diff:.4f}秒")
    print(f"每步时间比例: COPY是GAKL的 {avg_ratio:.2f}倍")
    
    if avg_diff > 0:
        print(f"  → COPY方法每步比GAKL慢 {avg_diff:.4f}秒 ({avg_diff/gakl_avg*100:.1f}%)")
    else:
        print(f"  → COPY方法每步比GAKL快 {abs(avg_diff):.4f}秒 ({abs(avg_diff)/gakl_avg*100:.1f}%)")
    
    # 最值对比
    print("\n【最值对比】")
    print(f"GAKL方法 - 最快: {gakl_stats['min_step_time_seconds']:.4f}秒, 最慢: {gakl_stats['max_step_time_seconds']:.4f}秒")
    print(f"COPY方法 - 最快: {copy_stats['min_step_time_seconds']:.4f}秒, 最慢: {copy_stats['max_step_time_seconds']:.4f}秒")
    
    # 从开始到结束的耗时对比
    print("\n【总耗时对比（包含所有开销）】")
    gakl_elapsed = gakl_stats['elapsed_time_seconds']
    copy_elapsed = copy_stats['elapsed_time_seconds']
    elapsed_diff = copy_elapsed - gakl_elapsed
    elapsed_ratio = copy_elapsed / gakl_elapsed if gakl_elapsed > 0 else 0
    
    print(f"GAKL方法总耗时: {gakl_elapsed:.2f}秒 ({gakl_elapsed/60:.2f}分钟, {gakl_elapsed/3600:.2f}小时)")
    print(f"COPY方法总耗时: {copy_elapsed:.2f}秒 ({copy_elapsed/60:.2f}分钟, {copy_elapsed/3600:.2f}小时)")
    print(f"耗时差异: {elapsed_diff:.2f}秒 ({elapsed_diff/60:.2f}分钟)")
    print(f"耗时比例: COPY是GAKL的 {elapsed_ratio:.2f}倍")
    
    # 计算开销分析
    print("\n【计算开销分析】")
    gakl_overhead = gakl_elapsed - gakl_total
    copy_overhead = copy_elapsed - copy_total
    
    print(f"GAKL方法额外开销: {gakl_overhead:.2f}秒 ({gakl_overhead/gakl_elapsed*100:.1f}%)")
    print(f"COPY方法额外开销: {copy_overhead:.2f}秒 ({copy_overhead/copy_elapsed*100:.1f}%)")
    
    # 总结
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    print(f"1. 每步训练时间: COPY方法 {'慢' if avg_diff > 0 else '快'} {abs(avg_diff):.4f}秒 ({abs(avg_diff)/gakl_avg*100:.1f}%)")
    print(f"2. 总训练时间: COPY方法 {'慢' if time_diff > 0 else '快'} {abs(time_diff):.2f}秒 ({abs(time_diff)/gakl_total*100:.1f}%)")
    print(f"3. 总耗时: COPY方法 {'慢' if elapsed_diff > 0 else '快'} {abs(elapsed_diff):.2f}秒 ({abs(elapsed_diff)/gakl_elapsed*100:.1f}%)")
    print("="*80)

def main():
    if len(sys.argv) < 3:
        print("用法: python compare_training_time.py <gakl_stats.json> <copy_stats.json>")
        print("\n示例:")
        print("  python compare_training_time.py \\")
        print("    /data1/zzj/llava_output_zzj_new/output_ga_joebiden/training_time_stats.json \\")
        print("    /data1/zzj/llava_output_zzj_new/output_sgpvt_joebiden/training_time_stats.json")
        sys.exit(1)
    
    gakl_file = sys.argv[1]
    copy_file = sys.argv[2]
    
    print(f"加载GAKL统计文件: {gakl_file}")
    gakl_stats = load_stats(gakl_file)
    if gakl_stats is None:
        sys.exit(1)
    
    print(f"加载COPY统计文件: {copy_file}")
    copy_stats = load_stats(copy_file)
    if copy_stats is None:
        sys.exit(1)
    
    compare_stats(gakl_stats, copy_stats)

if __name__ == "__main__":
    main()

