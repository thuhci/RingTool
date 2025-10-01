#!/bin/bash
# SpO2 (Oxygen Saturation) Experiments
# Ring1 & Ring2 - All Models

cd /root/ringtool_update_official_code_analysis

echo "=========================================="
echo "🩸 Starting SpO2 Experiments"
echo "=========================================="

# Ring1 SpO2 Experiments
echo ""
echo "📊 [1/8] Ring1 - ResNet - SpO2"
python main.py --config config/supervised/ring1/spo2/resnet-ring1-spo2-spo2-irred.json --data-path ../autodl-tmp/rings

echo ""
echo "📊 [2/8] Ring1 - InceptionTime - SpO2"
python main.py --config config/supervised/ring1/spo2/inceptiontime-ring1-spo2-spo2-irred.json --data-path ../autodl-tmp/rings

echo ""
echo "📊 [3/8] Ring1 - Transformer - SpO2"
python main.py --config config/supervised/ring1/spo2/transformer-ring1-spo2-spo2-irred.json --data-path ../autodl-tmp/rings

echo ""
echo "📊 [4/8] Ring1 - Mamba - SpO2"
python main.py --config config/supervised/ring1/spo2/mamba-ring1-spo2-spo2-irred.json --data-path ../autodl-tmp/rings

# Ring2 SpO2 Experiments
echo ""
echo "📊 [5/8] Ring2 - ResNet - SpO2"
python main.py --config config/supervised/ring2/spo2/resnet-ring2-spo2-spo2-irred.json --data-path ../autodl-tmp/rings

echo ""
echo "📊 [6/8] Ring2 - InceptionTime - SpO2"
python main.py --config config/supervised/ring2/spo2/inceptiontime-ring2-spo2-spo2-irred.json --data-path ../autodl-tmp/rings

echo ""
echo "📊 [7/8] Ring2 - Transformer - SpO2"
python main.py --config config/supervised/ring2/spo2/transformer-ring2-spo2-spo2-irred.json --data-path ../autodl-tmp/rings

echo ""
echo "📊 [8/8] Ring2 - Mamba - SpO2"
python main.py --config config/supervised/ring2/spo2/mamba-ring2-spo2-spo2-irred.json --data-path ../autodl-tmp/rings

echo ""
echo "=========================================="
echo "🎉 SpO2 Experiments Completed!"
echo "=========================================="
