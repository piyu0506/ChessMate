import pygame
import chess
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

# ==========================================
# 1. SETUP & PATHS
# ==========================================
# Window Settings
WIDTH, HEIGHT = 640, 640
SQ_SIZE = WIDTH // 8
FPS = 60

# Colors
LIGHT_SQ = (240, 217, 181)  # Wood Light
DARK_SQ = (181, 136, 99)    # Wood Dark
HIGHLIGHT = (255, 255, 0)   # Bright Yellow for selection
TEXT_COLOR = (0, 0, 0)

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, 'models'))
sys.path.append(os.path.join(project_root, 'src'))
ASSET_DIR = os.path.join(project_root, 'assets')

# Import Model
try:
    from model import ChessNet
    from encoder import encode_move
except ImportError:
    print("❌ Error: 'model.py' or 'encoder.py' missing.")
    sys.exit(1)

MODEL_PATH = os.path.join(project_root, 'models', 'chess_model.pth')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. AI & LOGIC
# ==========================================
def board_to_tensor(board):
    """ Converts board to (12, 8, 8) tensor. """
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    piece_map = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
                 "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11}
    for square, piece in board.piece_map().items():
        rank, file = divmod(square, 8)
        if piece.symbol() in piece_map:
            tensor[piece_map[piece.symbol()], rank, file] = 1.0
    return torch.from_numpy(tensor).unsqueeze(0).to(DEVICE)

def get_ai_move(model, board):
    model.eval()
    with torch.no_grad():
        tensor = board_to_tensor(board)
        policy_logits, value = model(tensor)
    
    # Flatten outputs
    policy_probs = F.softmax(policy_logits.flatten(), dim=0).cpu().numpy()
    
    legal_moves = list(board.legal_moves)
    move_scores = []

    # Score every legal move
    for move in legal_moves:
        idx = encode_move(move, board.turn)
        score = 0
        if idx is not None and idx < len(policy_probs):
            score = policy_probs[idx]
        move_scores.append((move, score))
    
    # Sort by score (descending)
    move_scores.sort(key=lambda x: x[1], reverse=True)
    
    # --- DEBUG: Print what the AI is thinking ---
    print(f"\n🧠 AI Thought Process (Value: {value.item():.2f}):")
    for i in range(min(3, len(move_scores))):
        m, s = move_scores[i]
        print(f"   {i+1}. {m.uci()} (Conf: {s*100:.1f}%)")
        
    # Pick best move (or random if confused)
    if not move_scores:
        return np.random.choice(legal_moves)
    
    return move_scores[0][0]

# ==========================================
# 3. GRAPHICS ENGINE
# ==========================================
class ChessGUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(f"Chess AI - Running on {DEVICE}")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 36, bold=True)
        
        self.board = chess.Board()
        self.selected_sq = None
        self.ai_thinking = False
        self.game_over = False
        
        # Load Model
        self.model = ChessNet(num_res_blocks=6).to(DEVICE)
        if os.path.exists(MODEL_PATH):
            try:
                self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
                print("✅ Model loaded successfully.")
            except:
                print("⚠️ Model architecture mismatch. Check num_res_blocks.")
        else:
            print("⚠️ No model file found.")

        # LOAD IMAGES (Auto-Scanner)
        self.images = {}
        self.load_images()

    def load_images(self):
        pieces = ['wp', 'wn', 'wb', 'wr', 'wq', 'wk', 'bp', 'bn', 'bb', 'br', 'bq', 'bk']
        for p in pieces:
            path_png = os.path.join(ASSET_DIR, f"{p}.png")
            if os.path.exists(path_png):
                img = pygame.image.load(path_png)
                self.images[p] = pygame.transform.scale(img, (SQ_SIZE, SQ_SIZE))
            # If missing, it will use text fallback automatically

    def draw_board(self):
        for row in range(8):
            for col in range(8):
                # Colors
                color = LIGHT_SQ if (row + col) % 2 == 0 else DARK_SQ
                
                # Highlight Selection
                sq_idx = (7 - row) * 8 + col
                if self.selected_sq == sq_idx:
                    color = HIGHLIGHT
                
                # Highlight Last Move
                if self.board.move_stack:
                    last = self.board.peek()
                    if sq_idx == last.from_square or sq_idx == last.to_square:
                         color = (200, 200, 50) # Yellowish tint

                pygame.draw.rect(self.screen, color, (col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))

    def draw_pieces(self):
        for square in range(64):
            piece = self.board.piece_at(square)
            if piece:
                color_code = 'w' if piece.color == chess.WHITE else 'b'
                key = f"{color_code}{piece.symbol().lower()}"
                
                row, col = 7 - (square // 8), (square % 8)
                x, y = col * SQ_SIZE, row * SQ_SIZE
                
                if key in self.images:
                    self.screen.blit(self.images[key], (x, y))
                else:
                    # Text Fallback
                    c = (0, 0, 0) if piece.color == chess.BLACK else (255, 255, 255)
                    txt = self.font.render(piece.symbol(), True, c)
                    self.screen.blit(txt, (x + 25, y + 15))

    def get_square(self, pos):
        x, y = pos
        return (7 - (y // SQ_SIZE)) * 8 + (x // SQ_SIZE)

    def run(self):
        running = True
        while running:
            self.draw_board()
            self.draw_pieces()
            if self.ai_thinking:
                pygame.draw.circle(self.screen, (255, 0, 0), (WIDTH-20, 20), 10)
            pygame.display.flip()
            self.clock.tick(FPS)

            # AI Move
            if not self.game_over and self.board.turn == chess.BLACK and not self.ai_thinking:
                self.ai_thinking = True
                # Redraw to show thinking indicator
                self.draw_board(); self.draw_pieces(); 
                pygame.draw.circle(self.screen, (255, 0, 0), (WIDTH-20, 20), 10)
                pygame.display.flip()
                
                move = get_ai_move(self.model, self.board)
                self.board.push(move)
                self.ai_thinking = False
                
                if self.board.is_game_over(): self.game_over = True

            # Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and not self.game_over:
                    if self.board.turn == chess.WHITE:
                        sq = self.get_square(event.pos)
                        if self.selected_sq is None:
                            p = self.board.piece_at(sq)
                            if p and p.color == chess.WHITE: self.selected_sq = sq
                        else:
                            move = chess.Move(self.selected_sq, sq)
                            # Auto-Queen
                            if self.board.piece_at(self.selected_sq).piece_type == chess.PAWN:
                                if (sq // 8) in [0, 7]: move.promotion = chess.QUEEN
                            
                            if move in self.board.legal_moves:
                                self.board.push(move)
                                self.selected_sq = None
                            else:
                                # Reselect
                                p = self.board.piece_at(sq)
                                if p and p.color == chess.WHITE: self.selected_sq = sq
                                else: self.selected_sq = None

        pygame.quit()

if __name__ == "__main__":
    gui = ChessGUI()
    gui.run()