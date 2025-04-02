# Reinforcement Learning chess-game / AnssiO

import pygame
import chess
import numpy as np
import torch  
import torch.nn as nn
import torch.optim as optim
import os
import random
import datetime

# Constants
WINDOW_SIZE = 600
SQUARE_SIZE = WINDOW_SIZE // 8
FPS = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 4096)  # 4096 = 64 squares * 64 squares (all possible moves)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 256 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ChessAgent:
    def __init__(self, device='cpu'): # or GPU
        self.device = device
        self.model = ChessNet().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.epsilon = 0.9  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def select_move(self, state, legal_moves):
        if random.random() < self.epsilon:
            return random.choice(list(legal_moves))
        
        # Ensure state is in correct format [batch_size, channels, height, width]
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Batch dimension
        state_tensor = state_tensor.to(self.device)
        
        with torch.no_grad():
            move_probs = self.model(state_tensor)
        
        # Filter only legal moves and select the best one
        legal_moves_list = list(legal_moves)
        if not legal_moves_list:
            return None
            
        legal_move_probs = [move_probs[0][self.move_to_index(move)] for move in legal_moves_list]
        best_move_idx = np.argmax(legal_move_probs)
        return legal_moves_list[best_move_idx]

    def move_to_index(self, move):
        from_square = move.from_square
        to_square = move.to_square
        return from_square * 64 + to_square

    def save_model(self, episode):
        # Create a filename with the current date and time
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"model_{timestamp}_episode_{episode}.pth"
        torch.save(self.model.state_dict(), model_filename)
        print(f"Model saved as: {model_filename}")

class ChessEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        self.clock = pygame.time.Clock()
        self.board = chess.Board()
        self.load_pieces()
        self.agent = ChessAgent()
        self.training = True
        self.move_count = 0

    def load_pieces(self):
        # Get the absolute path to the pieces directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pieces_dir = os.path.join(current_dir, 'pieces')
        
        print(f"Looking for pieces in: {pieces_dir}")
        
        # Dictionary to store piece images
        self.pieces = {}
        
        # Map chess piece symbols to file names
        piece_files = {
            'P': 'w_pawn_png_128px.png',
            'N': 'w_knight_png_128px.png',
            'B': 'w_bishop_png_128px.png',
            'R': 'w_rook_png_128px.png',
            'Q': 'w_queen_png_128px.png',
            'K': 'w_king_png_128px.png',
            'p': 'b_pawn_png_128px.png',
            'n': 'b_knight_png_128px.png',
            'b': 'b_bishop_png_128px.png',
            'r': 'b_rook_png_128px.png',
            'q': 'b_queen_png_128px.png',
            'k': 'b_king_png_128px.png'
        }
        
        for piece_symbol, filename in piece_files.items():
            try:
                piece_path = os.path.join(pieces_dir, filename)
                print(f"Loading piece from: {piece_path}")
                
                if not os.path.exists(piece_path):
                    raise FileNotFoundError(f"Piece not found: {piece_path}")
                
                self.pieces[piece_symbol] = pygame.transform.scale(
                    pygame.image.load(piece_path),
                    (SQUARE_SIZE, SQUARE_SIZE)
                )
            except Exception as e:
                print(f"Error loading piece {piece_symbol}: {str(e)}")
                raise

    def draw_board(self):
        for row in range(8):
            for col in range(8):
                color = WHITE if (row + col) % 2 == 0 else GRAY
                pygame.draw.rect(
                    self.screen,
                    color,
                    (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                )

    def draw_pieces(self):
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                x = chess.square_file(square) * SQUARE_SIZE
                y = (7 - chess.square_rank(square)) * SQUARE_SIZE
                self.screen.blit(self.pieces[piece.symbol()], (x, y))

    def render(self, episode, total_reward):
        self.draw_board()
        self.draw_pieces()

        # Display diagnostics
        stats_text = f"Episode: {episode}, Total Reward: {total_reward}, Moves: {self.move_count}"
        text_surface = pygame.font.SysFont('Arial', 24).render(stats_text, True, BLACK)
        self.screen.blit(text_surface, (10, 10))  # Positioning the text

        pygame.display.flip()
        self.clock.tick(FPS)

    def reset(self):
        self.board = chess.Board()
        return self.get_state()

    def get_state(self):
        # Convert chess board to numerical representation
        # Reshape to [12, 8, 8] format for PyTorch's CNN
        state = np.zeros((12, 8, 8), dtype=np.float32)  # Changed dimensions order
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                piece_type = piece.piece_type - 1
                if piece.color:  # White pieces
                    state[piece_type][rank][file] = 1
                else:  # Black pieces
                    state[piece_type + 6][rank][file] = 1
        return state

    def step(self, action):
        self.board.push(action)
        self.move_count += 1
        
        # Calculate reward
        reward = 0
        done = False
        
        if self.board.is_checkmate():
            reward = 100 if self.board.turn else -100
            done = True
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            reward = 0
            done = True
        elif self.board.is_check():
            reward = 5 if not self.board.turn else -5
        
        return self.get_state(), reward, done

    def train(self):
        state = self.get_state()
        done = False
        total_reward = 0
        
        while not done:
            legal_moves = list(self.board.legal_moves)
            if not legal_moves:
                break
                
            action = self.agent.select_move(state, legal_moves)
            next_state, reward, done = self.step(action)
            total_reward += reward
            
            self.render(0, 0)  # Placeholder for episode and total reward
            pygame.time.wait(100)  # Delay to make it visible
            
            state = next_state
            
        return total_reward

if __name__ == "__main__":
    env = ChessEnv()
    num_episodes = 1000 # Training episodes
    
    running = True
    episode = 0
    
    while running and episode < num_episodes:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Train one episode
        total_reward = env.train()
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")
        
        # Save the model after each episode
        env.agent.save_model(episode)
        
        # Reset for next episode
        env.reset()
        episode += 1
        
        # Decay epsilon
        env.agent.epsilon = max(
            env.agent.epsilon_min,
            env.agent.epsilon * env.agent.epsilon_decay
        )
        
        # Update render with episode and total reward
        env.render(episode, total_reward)

    pygame.quit()

