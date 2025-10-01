#!/bin/bash
# BP (Blood Pressure) Experiments
# Ring1 & Ring2 - All Models

cd /root/ringtool_update_official_code_analysis

echo "=========================================="
echo "🩺 Starting BP Experiments"
echo "=========================================="

# Ring1 BP Experiments
echo ""
echo "📊 [1/8] Ring1 - ResNet - BP"
python main.py --config config/supervised/ring1/bp/resnet-ring1-bp-all-irred.json --data-path ../autodl-tmp/rings

echo ""
echo "📊 [2/8] Ring1 - InceptionTime - BP"
python main.py --config config/supervised/ring1/bp/inceptiontime-ring1-bp-all-irred.json --data-path ../autodl-tmp/rings

echo ""
echo "📊 [3/8] Ring1 - Transformer - BP"
python main.py --config config/supervised/ring1/bp/transformer-ring1-bp-all-irred.json --data-path ../autodl-tmp/rings

echo ""
echo "📊 [4/8] Ring1 - Mamba - BP"
python main.py --config config/supervised/ring1/bp/mamba-ring1-bp-all-irred.json --data-path ../autodl-tmp/rings

# Ring2 BP Experiments
echo ""
echo "📊 [5/8] Ring2 - ResNet - BP"
python main.py --config config/supervised/ring2/bp/resnet-ring2-bp-all-irred.json --data-path ../autodl-tmp/rings

echo ""
echo "📊 [6/8] Ring2 - InceptionTime - BP"
python main.py --config config/supervised/ring2/bp/inceptiontime-ring2-bp-all-irred.json --data-path ../autodl-tmp/rings

echo ""
echo "📊 [7/8] Ring2 - Transformer - BP"
python main.py --config config/supervised/ring2/bp/transformer-ring2-bp-all-irred.json --data-path ../autodl-tmp/rings

echo ""
echo "📊 [8/8] Ring2 - Mamba - BP"
python main.py --config config/supervised/ring2/bp/mamba-ring2-bp-all-irred.json --data-path ../autodl-tmp/rings

echo ""
echo "=========================================="
echo "🎉 BP Experiments Completed!"
echo "=========================================="
