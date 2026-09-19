# -*- coding: utf-8 -*-
"""Better-1 复现工程公共配置层：统一从 config.json 读取路径与固定配置。

优先级：命令行参数 > 环境变量（BETTER1_*）> config.json 默认值
"""
import argparse
import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "config.json")


def load_config(path=None):
    """读取配置文件；未指定时使用工程根目录 config.json（可用环境变量 BETTER1_CONFIG 覆盖）。"""
    cfg_path = path or os.environ.get("BETTER1_CONFIG") or DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _abspath(p):
    return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)


def resolve_dirs(cfg, args=None):
    """解析输出目录与中间产物目录（相对路径以工程根目录为基准），并自动创建。"""
    out = getattr(args, "output_dir", None) or os.environ.get("BETTER1_OUTPUT_DIR") or cfg["paths"]["output_dir"]
    tmp = getattr(args, "temp_dir", None) or os.environ.get("BETTER1_TEMP_DIR") or cfg["paths"]["temp_dir"]
    out, tmp = _abspath(out), _abspath(tmp)
    os.makedirs(out, exist_ok=True)
    os.makedirs(tmp, exist_ok=True)
    return out, tmp


def resolve_source(cfg, args=None):
    """解析临床主表路径（E0 输入，只读）。"""
    return (getattr(args, "source", None)
            or os.environ.get("BETTER1_CLINICAL_SOURCE")
            or cfg["paths"]["clinical_source"])


def build_parser(prog, description=""):
    """构造带通用参数的命令行解析器；脚本可继续 add_argument 追加专属参数。"""
    p = argparse.ArgumentParser(prog=prog, description=description)
    p.add_argument("--config", default=None, help="config.json 路径（默认使用工程根目录下的 config.json）")
    p.add_argument("--source", default=None, help="临床主表 xlsx 绝对路径覆盖（仅 E0 使用）")
    p.add_argument("--output-dir", default=None, help="输出目录覆盖（默认 <工程根>/results）")
    p.add_argument("--temp-dir", default=None, help="中间产物目录覆盖（默认 <工程根>/temp_internal）")
    return p


def require_file(path, hint=""):
    """输入文件存在性检查，缺失时给出明确指引而非抛出难解异常。"""
    if not os.path.exists(path):
        raise SystemExit("[ERROR] 输入数据不存在：%s\n%s" % (path, hint))
    return path
