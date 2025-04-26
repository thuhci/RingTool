#!/usr/bin/env bash
set -euo pipefail

chmod +x  ./run.sh
# 用法: ./run_all.sh [config_root_dir]
# 如果不传参数，则默认在当前目录查找
if [[ $# -gt 1 ]]; then
  echo "用法: $0 [配置根目录]" >&2
  exit 1
fi

ROOT_DIR="${1:-.}"

# 开启递归 glob 和空匹配自动跳过
shopt -s globstar nullglob

# 保存当前脚本执行目录
SCRIPT_DIR=$(pwd)

# 进入根目录，方便使用相对路径
pushd "$ROOT_DIR" > /dev/null

# 收集所有 .json 文件到数组
configs=(**/*.json)

if [[ ${#configs[@]} -eq 0 ]]; then
  echo "在 '$ROOT_DIR' 下未找到任何 .json 文件。" >&2
  popd > /dev/null
  exit 1
fi

# 将相对路径转换为绝对路径
abs_configs=()
for f in "${configs[@]}"; do
  abs_configs+=("$(realpath "$f")")
done

# 遍历每个配置文件，单独调用 main.py
for config_file in "${abs_configs[@]}"; do
  echo "处理配置文件: $config_file"
  python "$SCRIPT_DIR/main.py" --config "$config_file"
done

# 返回原来目录
popd > /dev/null