# RL-Chess: Reinforcement Learning Chess Game

A chess game implementation that uses reinforcement learning to train an AI agent to play chess. The project combines PyTorch for the neural network, python-chess for the chess engine, and Pygame for the visual interface.

## Features

- Visual chess board interface using Pygame
- Deep neural network-based chess agent
- Reinforcement learning implementation
- Real-time training visualization
- Automatic model saving after each episode
- Epsilon-greedy exploration strategy

## Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (optional, but recommended for faster training)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/a0w3b/RL-CHESS.git
cd RL-Chess
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
OR

```bash
conda create -n rl-chess pyython=3.10
conda activate rl-chess
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```
## Test
```bash
python -c "import pygame; import chess; import torch; print('All packages installed successfully!')"
```

## Project Structure

- `RL-Chess.py`: Main game and training implementation
- `pieces/`: Directory containing chess piece images
- `requirements.txt`: Project dependencies
- `README.md`: Project documentation

## Usage

Run the main script to start training:
```bash
python RL-Chess.py
```

The program will:
1. Initialize the chess environment and neural network
2. Start training episodes (default: 1000 episodes)
3. Display the training progress in real-time
4. Save model checkpoints after each episode

## Training Details

- The neural network uses a CNN architecture with 3 convolutional layers
- Training uses an epsilon-greedy exploration strategy
- Epsilon decay rate: 0.995
- Minimum epsilon: 0.01
- Learning rate: 0.001

## Model Architecture

The chess agent uses a CNN with the following structure:
- Input: 12 channels (6 piece types × 2 colors)
- Conv1: 64 filters
- Conv2: 128 filters
- Conv3: 256 filters
- FC1: 1024 neurons
- Output: 4096 neurons (64×64 possible moves)

## License

This project is open source and available under the MIT License.

## Author

AnssiO 