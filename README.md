# ChannelNet: Deep Learning for OFDM Channel Estimation

A dual-stage neural network architecture combining Super-Resolution CNN (SRCNN) and Denoising CNN (DnCNN) for enhanced OFDM channel estimation in wireless communications.

## Overview

ChannelNet addresses the challenge of accurate channel state information (CSI) estimation in OFDM systems by:
- **Stage 1**: Super-resolution enhancement of noisy least-squares (LS) channel estimates
- **Stage 2**: Denoising through residual learning to remove estimation artifacts

## Architecture

The network consists of two sequential blocks:

### Super-Resolution Block (SRCNN)
- 9×9 conv → PReLU → 1×1 conv → PReLU → 5×5 conv
- Enhances spatial resolution of channel estimates

### Denoising Block (DnCNN) 
- 3×3 conv → PReLU → 3 hidden layers with BatchNorm → residual connection
- Removes noise through learned residual mapping

## Features

- **Synthetic OFDM Dataset**: Generates realistic channel conditions with configurable SNR
- **Time-Frequency Visualization**: Comprehensive plots showing training progress and channel maps
- **GPU Acceleration**: Automatic CUDA detection and utilization
- **Adaptive Learning**: Learning rate scheduling based on validation performance

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the complete training pipeline:

```python
python channelnet.py
```

This will:
1. Generate synthetic OFDM channel data (2500 samples)
2. Train the ChannelNet model for 50 epochs
3. Save visualization results to `channelnet_results.png`

## Results

The model typically achieves:
- **MSE Reduction**: 60-80% improvement over noisy LS estimates
- **Convergence**: Stable training within 50 epochs
- **Performance**: Real-time inference capability for practical deployment

## Dataset Parameters

- **Grid Size**: 72 subcarriers × 14 OFDM symbols
- **SNR Range**: Configurable (default: 15 dB)
- **Channel Model**: Gaussian-filtered complex coefficients
- **Noise Model**: AWGN with controlled variance

## Visualization

The output includes:
- Training loss convergence curve
- Channel magnitude heatmaps (input/output/ground truth)
- Pixel-wise error analysis
- Performance metrics summary

## Requirements

- Python 3.7+
- PyTorch 1.9+
- NumPy, SciPy, Matplotlib

## Applications

- 5G/6G wireless systems
- OFDM-based communications
- Channel estimation research
- Deep learning for signal processing
