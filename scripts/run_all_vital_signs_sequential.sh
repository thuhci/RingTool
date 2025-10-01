#!/bin/bash
# Sequential Execution of All Vital Signs Experiments
# Runs RR -> SpO2 -> BP in order (one after another)

cd /root/ringtool_update_official_code_analysis

echo "=========================================="
echo "🚀 顺序运行所有生理指标实验"
echo "=========================================="

# 创建日志目录
mkdir -p logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 1. RR实验
echo ""
echo "🫁 [阶段 1/3] 开始 RR 实验..."
./scripts/run_rr_experiments.sh 2>&1 | tee logs/rr_experiments_${TIMESTAMP}.log

# 2. SpO2实验
echo ""
echo "🩸 [阶段 2/3] 开始 SpO2 实验..."
./scripts/run_spo2_experiments.sh 2>&1 | tee logs/spo2_experiments_${TIMESTAMP}.log

# 3. BP实验
echo ""
echo "🩺 [阶段 3/3] 开始 BP 实验..."
./scripts/run_bp_experiments.sh 2>&1 | tee logs/bp_experiments_${TIMESTAMP}.log

echo ""
echo "=========================================="
echo "🎉 所有实验完成！"
echo "=========================================="
echo "📊 日志文件:"
echo "  - RR:   logs/rr_experiments_${TIMESTAMP}.log"
echo "  - SpO2: logs/spo2_experiments_${TIMESTAMP}.log"
echo "  - BP:   logs/bp_experiments_${TIMESTAMP}.log"
echo "=========================================="
