import pygame
import chess
import torch
import os
import sys
import threading

# --- 1. SETUP PATHS & IMPORTS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # python/
sys.path.append(current_dir)

try:
    from model import ChessNet
    from mcts import MCTS
except ImportError:
    print("CRITICAL ERROR: Could not import 'model' or 'mcts'.")
    print("Make sure model.py and mcts.py are in the same folder as play.py")
    sys.exit(1)

# --- 2. CONFIGURATION ---
WIDTH, HEIGHT = 640, 640
SQ_SIZE = WIDTH // 8
ASSET_PATH = os.path.join(project_root, "assets")
MODEL_PATH = os.path.join(project_root, "models", "chess_model.pth")

# --- 3. GRAPHICS HELPERS ---
def load_piece_images():
    pieces = ['wp', 'wn', 'wb', 'wr', 'wq', 'wk', 'bp', 'bn', 'bb', 'br', 'bq', 'bk']
    images = {}
    for p in pieces:
        img_path = os.path.join(ASSET_PATH, f"{p}.png")
        if os.path.exists(img_path):
            images[p] = pygame.transform.scale(pygame.image.load(img_path), (SQ_SIZE, SQ_SIZE))
        else:
            # Fallback square if image missing
            s = pygame.Surface((SQ_SIZE, SQ_SIZE))
            s.fill((200, 50, 50)) 
            images[p] = s
    return images

def draw_board(screen, board, selected_sq, piece_images):
    # Draw Grid
    for r in range(8):
        for c in range(8):
            color = (238, 238, 210) if (r + c) % 2 == 0 else (118, 150, 86)
            pygame.draw.rect(screen, color, (c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))
    
    # Highlight Selection
    if selected_sq is not None:
        r, c = 7 - chess.square_rank(selected_sq), chess.square_file(selected_sq)
        s = pygame.Surface((SQ_SIZE, SQ_SIZE))
        s.set_alpha(100)
        s.fill((255, 255, 0))
        screen.blit(s, (c*SQ_SIZE, r*SQ_SIZE))

    # Highlight Last Move
    if board.move_stack:
        last_move = board.peek()
        for sq in [last_move.from_square, last_move.to_square]:
            r, c = 7 - chess.square_rank(sq), chess.square_file(sq)
            s = pygame.Surface((SQ_SIZE, SQ_SIZE))
            s.set_alpha(100)
            s.fill((155, 155, 255))
            screen.blit(s, (c*SQ_SIZE, r*SQ_SIZE))

    # Draw Pieces
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            color = 'w' if p.color == chess.WHITE else 'b'
            key = color + p.symbol().lower()
            x = chess.square_file(sq) * SQ_SIZE
            y = (7 - chess.square_rank(sq)) * SQ_SIZE
            screen.blit(piece_images[key], (x, y))

# --- 4. MAIN LOOP ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("AlphaZero Engine (MCTS)")
    clock = pygame.time.Clock()
    piece_images = load_piece_images()
    
    board = chess.Board()
    
    # --- LOAD BRAIN ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Engine running on: {device}")

    try:
        model = ChessNet(num_res_blocks=6).to(device)
        # Load weights safely
        if os.path.exists(MODEL_PATH):
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            print("Model Loaded Successfully.")
        else:
            print(f"WARNING: No model found at {MODEL_PATH}. AI will be random/dumb.")
        
        # Initialize MCTS (The High Level Logic)
        mcts_engine = MCTS(model, device, simulations=600)

    except Exception as e:
        print(f"Error loading model: {e}")
        return

    selected_sq = None
    running = True
    ai_thinking = False
    ai_result = [None]   # shared container: thread writes here, main thread reads
    t = None             # handle to the current AI thread

    while running:
        # 1. CHECK GAME OVER (Fixes your crash)
        if board.is_game_over():
            draw_board(screen, board, selected_sq, piece_images)
            
            # Draw Game Over Text
            font = pygame.font.SysFont('Arial', 32, bold=True)
            text = font.render(f"Game Over: {board.result()}", True, (255, 0, 0))
            box = pygame.Surface((400, 80))
            box.set_alpha(200); box.fill((255, 255, 255))
            screen.blit(box, (WIDTH//2 - 200, HEIGHT//2 - 40))
            screen.blit(text, text.get_rect(center=(WIDTH//2, HEIGHT//2)))
            
            pygame.display.flip()
            
            # Wait for quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
            continue

        # 2. AI TURN — launch search on a background thread
        if not board.turn and not ai_thinking:  # Black to move
            ai_thinking = True
            ai_result[0] = None
            # Force redraw before thinking so the screen shows the latest state
            draw_board(screen, board, selected_sq, piece_images)
            pygame.display.flip()

            board_copy = board.copy()  # give the thread its own snapshot

            def ai_move(b):
                try:
                    ai_result[0] = mcts_engine.search(b)  # works on copy only
                except Exception as e:
                    print(f"AI Error: {e}")

            t = threading.Thread(target=ai_move, args=(board_copy,), daemon=True)
            t.start()

        # 2b. Collect AI result on the main thread once the thread is done
        if ai_thinking and t is not None and not t.is_alive():
            if ai_result[0] is not None:
                board.push(ai_result[0])
                # print(f"AI Played: {ai_result[0]}")
            ai_thinking = False

        # 3. HUMAN INPUT
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if board.turn and event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                col = pos[0] // SQ_SIZE
                row = pos[1] // SQ_SIZE
                sq = chess.square(col, 7 - row)

                if selected_sq is None:
                    p = board.piece_at(sq)
                    if p and p.color == chess.WHITE:
                        selected_sq = sq
                else:
                    move = chess.Move(selected_sq, sq)
                    # Auto-Queen
                    if chess.Move(selected_sq, sq, promotion=chess.QUEEN) in board.legal_moves:
                        move = chess.Move(selected_sq, sq, promotion=chess.QUEEN)
                    
                    if move in board.legal_moves:
                        board.push(move)
                        selected_sq = None
                    else:
                        # Clicked elsewhere? Reselect
                        p = board.piece_at(sq)
                        if p and p.color == chess.WHITE: selected_sq = sq
                        else: selected_sq = None

        # 4. RENDER
        draw_board(screen, board, selected_sq, piece_images)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()