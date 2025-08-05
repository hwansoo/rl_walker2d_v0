#!/bin/bash

echo "Walker2D Bump Challenge Training Script"
echo "======================================"

# Create necessary directories
mkdir -p checkpoints/bump_practice
mkdir -p checkpoints/bump_challenge  
mkdir -p checkpoints/curriculum_learning
mkdir -p logs

echo "Training options:"
echo "1. Bump Practice (easier training)"
echo "2. Bump Challenge (direct challenge training)"
echo "3. Transfer Learning (start from pretrained model)"
echo "4. Curriculum Learning (progressive difficulty)"

echo ""
echo "Recommended training sequence for maximum performance:"
echo ""

echo "Option 1: Transfer learning (uses original observation space + enhanced rewards)"
echo "Step 1: Transfer learning on bump practice"
echo "Command: python learning.py --bump_practice --transfer_learning"
echo ""
echo "Step 2: Transfer learning on bump challenge"  
echo "Command: python learning.py --bump_challenge --transfer_learning"
echo ""

echo "Option 2: Training from scratch with enhanced observations"
echo "Command: python learning.py --bump_challenge --enhanced_obs"
echo ""

echo "Option 3: Curriculum learning"
echo "Command: python learning.py --curriculum --transfer_learning"
echo ""

echo "Note: Transfer learning uses original observation space for compatibility"
echo "      Enhanced observations (--enhanced_obs) only work when training from scratch"
echo ""

echo "Monitor training progress with TensorBoard:"
echo "tensorboard --logdir ./logs"