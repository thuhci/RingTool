#!/bin/bash
# RR (Respiratory Rate) Experiments
# Ring1 & Ring2 - All Models (8 experiments total)

cd /root/ringtool_update_official_code_analysis

echo "=========================================="
echo "🫁 Starting RR Experiments"
echo "=========================================="

# Ring1 RR Experiments
echo ""
echo "📊 [1/8] Ring1 - ResNet - RR"
python main.py --config config/supervised/ring1/rr/ir/resnet-ring1-rr-all-ir.json --data-path ../autodl-tmp/rings

echo ""
echo "📊 [2/8] Ring1 - InceptionTime - RR"
python main.py --config config/supervised/ring1/rr/ir/inception-time-ring1-rr-all-ir.json --data-path ../autodl-tmp/rings

echo ""
echo "📊 [3/8] Ring1 - Transformer - RR"
python main.py --config config/supervised/ring1/rr/ir/transformer-ring1-rr-all-ir.json --data-path ../autodl-tmp/rings

echo ""
echo "📊 [4/8] Ring1 - Mamba2 - RR"
python main.py --config config/supervised/ring1/rr/ir/mamba2-ring1-rr-all-ir.json --data-path ../autodl-tmp/rings

# Ring2 RR Experiments
echo ""
echo "📊 [5/8] Ring2 - ResNet - RR"
python main.py --config config/supervised/ring2/rr/ir/resnet-ring2-rr-all-ir.json --data-path ../autodl-tmp/rings

echo ""
echo "📊 [6/8] Ring2 - InceptionTime - RR"
python main.py --config config/supervised/ring2/rr/ir/inception-time-ring2-rr-all-ir.json --data-path ../autodl-tmp/rings

echo ""
echo "📊 [7/8] Ring2 - Transformer - RR"
python main.py --config config/supervised/ring2/rr/ir/transformer-ring2-rr-all-ir.json --data-path ../autodl-tmp/rings

echo ""
echo "📊 [8/8] Ring2 - Mamba2 - RR"
python main.py --config config/supervised/ring2/rr/ir/mamba2-ring2-rr-all-ir.json --data-path ../autodl-tmp/rings

echo ""
echo "=========================================="
echo "🎉 RR Experiments Completed!"
echo "=========================================="