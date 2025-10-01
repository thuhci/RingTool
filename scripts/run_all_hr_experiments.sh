#!/bin/bash
# HR实验批量训练脚本

cd /root/RingTool

echo "🚀 Starting all HR experiments..."

# Ring1 experiments
echo "📊 Ring1 - ResNet"
python main.py --config config/supervised/ring1/hr/ir/resnet-ring1-hr-all-ir.json --data-path ../autodl-tmp/rings

echo "📊 Ring1 - InceptionTime"
python main.py --config config/supervised/ring1/hr/ir/inception-time-ring1-hr-all-ir.json --data-path ../autodl-tmp/rings

echo "📊 Ring1 - Transformer"
python main.py --config config/supervised/ring1/hr/ir/transformer-ring1-hr-all-ir.json --data-path ../autodl-tmp/rings

echo "📊 Ring1 - Mamba2"
python main.py --config config/supervised/ring1/hr/ir/mamba2-ring1-hr-all-ir.json --data-path ../autodl-tmp/rings

# Ring2 experiments
echo "📊 Ring2 - ResNet"
python main.py --config config/supervised/ring2/hr/ir/resnet-ring2-hr-all-ir.json --data-path ../autodl-tmp/rings

echo "📊 Ring2 - InceptionTime"
python main.py --config config/supervised/ring2/hr/ir/inception-time-ring2-hr-all-ir.json --data-path ../autodl-tmp/rings

echo "📊 Ring2 - Transformer"
python main.py --config config/supervised/ring2/hr/ir/transformer-ring2-hr-all-ir.json --data-path ../autodl-tmp/rings

echo "📊 Ring2 - Mamba2"
python main.py --config config/supervised/ring2/hr/ir/mamba2-ring2-hr-all-ir.json --data-path ../autodl-tmp/rings

echo "🎉 All experiments completed!"