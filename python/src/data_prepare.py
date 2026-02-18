import chess.pgn
import numpy as np
import os
import sys
from tqdm import tqdm

# Add current dir to path to find encoder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from encoder import encode_move

# --- CONFIG ---
PGN_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "games.pgn")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TARGET_POSITIONS = 500000

def board_to_tensor(board):
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    piece_map = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
                 "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11}
    for square, piece in board.piece_map().items():
        rank, file = divmod(square, 8)
        tensor[piece_map[piece.symbol()], rank, file] = 1.0
    return tensor.flatten()

def process_pgn():
    print(f"Processing {PGN_FILE}...")
    X, y, z = [], [], []
    count = 0
    
    with open(PGN_FILE, encoding="utf-8", errors="replace") as pgn:
        pbar = tqdm(total=TARGET_POSITIONS)
        while count < TARGET_POSITIONS:
            try: game = chess.pgn.read_game(pgn)
            except: continue
            if game is None: break
            
            # Filter weak games
            w = game.headers.get("WhiteElo", "0")
            b = game.headers.get("BlackElo", "0")
            if not w.isdigit() or not b.isdigit() or int(w) < 2000 or int(b) < 2000: continue
            
            res = game.headers.get("Result", "*")
            root_val = 1.0 if res == "1-0" else (-1.0 if res == "0-1" else 0.0)
            if root_val == 0.0: continue # Skip draws for sharper training

            board = game.board()
            for move in game.mainline_moves():
                X.append(board_to_tensor(board))
                y.append(encode_move(move, board.turn))
                
                # Z is relative to current player
                current_val = root_val if board.turn == chess.WHITE else -root_val
                z.append(current_val)
                
                board.push(move)
                count += 1
                pbar.update(1)
                if count >= TARGET_POSITIONS: break
    
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), np.array(X, dtype=np.float32))
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), np.array(y, dtype=np.int16))
    np.save(os.path.join(OUTPUT_DIR, "z_train.npy"), np.array(z, dtype=np.float32))
    print("Done.")

if __name__ == "__main__":
    process_pgn()