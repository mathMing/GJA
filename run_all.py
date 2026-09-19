# -*- coding: utf-8 -*-
"""Better-1 E0/E1 复现工程统一运行入口。

用法：
    python run_all.py                      # 依次运行 E0 与 E1（发表口径：E1 REPS=200, N=20000）
    python run_all.py --only e0            # 只跑 E0（数据审计复算）
    python run_all.py --only e1            # 只跑 E1（合成机制实验）
    python run_all.py --quick              # 快速自检（E1 降为 REPS=3, N=3000），用于验证环境
    python run_all.py --source D:/x.xlsx --output-dir D:/out   # 覆盖数据源与输出目录

产物：
    <工程根>/results/            输出文件（xlsx / png）
    <工程根>/temp_internal/      中间产物（csv / json）
"""
import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")


def run(script_name, extra_args):
    cmd = [sys.executable, os.path.join(SRC_DIR, script_name)] + extra_args
    print("\n[RUN] " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=SRC_DIR)
    if proc.returncode != 0:
        print(f"[FAIL] {script_name} 退出码 {proc.returncode}")
    return proc.returncode


def main():
    p = argparse.ArgumentParser(prog="run_all", description="Better-1 E0/E1 复现工程运行入口")
    p.add_argument("--only", choices=["e0", "e1", "synthetic", "three_way", "report", "all"], default="all", help="选择运行阶段")
    p.add_argument("--quick", action="store_true", help="快速自检（仅影响 E1 规模），非发表口径")
    p.add_argument("--config", default=None, help="config.json 路径覆盖")
    p.add_argument("--source", default=None, help="临床主表 xlsx 路径覆盖（E0 使用）")
    p.add_argument("--output-dir", default=None, help="输出目录覆盖")
    p.add_argument("--temp-dir", default=None, help="中间产物目录覆盖")
    args = p.parse_args()

    common = []
    for flag, val in [("--config", args.config), ("--source", args.source),
                      ("--output-dir", args.output_dir), ("--temp-dir", args.temp_dir)]:
        if val:
            common += [flag, val]

    codes = []
    if args.only in ("e0", "all"):
        codes.append(run("e0_audit.py", common))
    if args.only in ("e1", "all"):
        extra = list(common)
        if args.quick:
            extra.append("--quick")
        codes.append(run("e1_synthetic.py", extra))

    if args.only in ('three_way', 'all'):
        codes.append(run('three_way_experiment.py', common))
    if args.only in ('synthetic', 'all'):
        codes.append(run('synthetic_mechanisms.py', common))
    if args.only in ('report', 'all'):
        codes.append(run('report_results.py', common))

    if any(c != 0 for c in codes):
        sys.exit(1)
    print("\n[DONE] 全部阶段执行完成。")


if __name__ == "__main__":
    main()


