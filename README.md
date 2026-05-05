# JAX DrQ Implementation

A JAX reimplementation of **DrQ (Data-regularized Q-learning)** with Comet ML logging.

## Requirements

**Python Version: 3.10**

## Overview

This is an unofficial reimplementation of DrQ using:
- **JAX** for neural networks and automatic differentiation
- **Optax** for optimization
- **Comet ML** for experiment tracking and logging
- **DMControl** for environments

Original paper: [Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels](https://arxiv.org/abs/2004.13649)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. For GPU support (recommended), install JAX with CUDA:
```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Usage

### Setup Comet ML

Set your Comet ML API key as an environment variable:
```bash
export COMET_API_KEY=your_api_key_here
```

Or edit `config.yaml` to include your API key directly.

### Training

Example of run:
```bash
# Edit config.yaml and set: use_augmentation: false
python train.py --experiment_name my_experiment --env cheetah_run --seed 42
```

### Available Environments

From DeepMind Control Suite:
- `cartpole_swingup`
- `cheetah_run`
- `walker_walk`
- `finger_spin`
- `reacher_easy`
- `ball_in_cup_catch`
- And many more...

### SAC vs DrQ

This implementation supports both algorithms with configurable regularization strength:

**DrQ (Data-regularized Q-learning)** - Full implementation with three regularization mechanisms:

1. **Image transformations**: Random crop with ±4 pixel shifts (padding + crop)
2. **Q-target averaging** (K): Average Q targets over K augmented versions (K=1 or K=2)
3. **Q-function averaging** (M): Train Q on M augmented observations (M=1 or M=2)

This significantly reduces overfitting and improves sample efficiency on pixel-based tasks.

## Implementation Details

### DrQ Regularization Mechanisms

**1. Random Crop Augmentation:**
```yaml
image_size: 84
image_pad: 4  # Pad each side by 4 pixels
```
- Pad image to 92×92 (repeat boundary pixels)
- Random crop back to 84×84
- Effective shift: ±4 pixels in each direction

**2. Q-Target Averaging (Parameter K):**
```python
# K=2 (default): Average over 2 augmentations
target_q_original = reward + discount * V(next_obs)
target_q_aug = reward + discount * V(next_obs_aug)
target_q = (target_q_original + target_q_aug) / 2

# K=1: Use only original (no averaging)
target_q = reward + discount * V(next_obs)
```

**3. Q-Function Averaging (Parameter M):**
```python
# M=2 (default): Train on both observations
loss = (Q(obs, action) - target_q)² + (Q(obs_aug, action) - target_q)²

# M=1: Train on original only
loss = (Q(obs, action) - target_q)²
```

**Configurable Regularization Strength:**

The combination of K and M determines regularization strength:
- **K=2, M=2** (Full DrQ): Maximum regularization, best for small datasets
- **K=2, M=1**: Strong target regularization, lighter computational cost  
- **K=1, M=1** (Simplified): Minimal regularization, faster training
- **K=1, M=2**: Heavier Q regularization without target averaging


## Project Structure

```
drq/
├── drq.py           # DrQ agent implementation
├── networks.py          # Neural network architectures
├── replay_buffer.py # Experience replay with augmentation
├── utils.py         # Utility functions
├── train.py         # Training script
├── config.yaml      # Configuration file
├── requirements.txt     # Python dependencies
└── README.md           # This file
```


## Acknowledgments

- Original PyTorch implementation: https://github.com/denisyarats/drq
- JAX/Flax implementation reference: https://github.com/ikostrikov/jax-rl
