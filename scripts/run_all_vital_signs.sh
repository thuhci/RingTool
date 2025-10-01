#!/bin/bash
# Master Script to Run All Vital Signs Experiments in Screen Sessions
# Creates separate screen sessions for RR, SpO2, and BP experiments

cd /root/ringtool_update_official_code_analysis

echo "=========================================="
echo "🚀 启动所有生理指标实验"
echo "=========================================="

# 检查screen是否安装
if ! command -v screen &> /dev/null; then
    echo "❌ Screen未安装，正在安装..."
    apt-get update && apt-get install -y screen
fi

# 1. RR实验
echo ""
echo "🫁 [1/3] 启动 RR 实验 (screen: rr_exp)"
screen -dmS rr_exp bash -c "cd /root/ringtool_update_official_code_analysis && ./scripts/run_rr_experiments.sh 2>&1 | tee logs/rr_experiments_$(date +%Y%m%d_%H%M%S).log"
echo "✅ RR实验已在后台启动 (screen -r rr_exp 查看)"

sleep 2

# 2. SpO2实验
echo ""
echo "🩸 [2/3] 启动 SpO2 实验 (screen: spo2_exp)"
screen -dmS spo2_exp bash -c "cd /root/ringtool_update_official_code_analysis && ./scripts/run_spo2_experiments.sh 2>&1 | tee logs/spo2_experiments_$(date +%Y%m%d_%H%M%S).log"
echo "✅ SpO2实验已在后台启动 (screen -r spo2_exp 查看)"

sleep 2

# 3. BP实验
echo ""
echo "🩺 [3/3] 启动 BP 实验 (screen: bp_exp)"
screen -dmS bp_exp bash -c "cd /root/ringtool_update_official_code_analysis && ./scripts/run_bp_experiments.sh 2>&1 | tee logs/bp_experiments_$(date +%Y%m%d_%H%M%S).log"
echo "✅ BP实验已在后台启动 (screen -r bp_exp 查看)"

echo ""
echo "=========================================="
echo "📋 Screen 会话列表:"
echo "=========================================="
screen -ls

echo ""
echo "📖 使用说明:"
echo "  - 查看RR实验:   screen -r rr_exp"
echo "  - 查看SpO2实验: screen -r spo2_exp"
echo "  - 查看BP实验:   screen -r bp_exp"
echo "  - 退出screen:   Ctrl+A 然后按 D"
echo "  - 查看日志:     tail -f logs/*_experiments_*.log"
echo ""
echo "🎉 所有实验已启动！"
echo "=========================================="
