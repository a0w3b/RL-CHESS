# Reinforcement Learning Chess Game
![RL-CHESS](/rl-chess02.png)

## Summary

This project, **RL-Chess**, is a comprehensive chess game implementation designed to train an AI agent using **reinforcement learning**. It leverages **PyTorch** for the deep neural network, the **python-chess** library for robust game logic, and **Pygame** for an interactive visual interface.

### Key Features:

- **Deep Reinforcement Learning Agent:** A Convolutional Neural Network (CNN) learns chess strategies through a reinforcement learning approach. It employs an **epsilon-greedy exploration** strategy, with epsilon decaying over episodes to balance exploring new moves and exploiting learned knowledge as training progresses.
    
- **Interactive Pygame Interface:** A clear and intuitive graphical representation of the chessboard allows for real-time visualization of gameplay and agent training.
    
- **Real-time Training Visualization:** Observe the AI agent's progress directly on the chessboard as it learns through each episode.
    
- **Automatic Model Saving:** The trained neural network model is automatically saved after every episode, enabling progress tracking and the resumption of training.
    
- **Modular Design:** The project features separated components for the Chess Environment, Chess Agent, and the Neural Network architecture, promoting code clarity and maintainability.
  
![Flowchart](/RL-CHESS.png)

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
.

├── pieces/
|
├── README.md
|
├── RL-CHESS.png
|
├── RL-Chess.py
|
├── model_20250308_151925_episode_23.pth
|
├── requirements.txt
|
└── rl-chess02.png

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
