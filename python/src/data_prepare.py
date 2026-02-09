import chess.pgn
import numpy as np
import os
from tqdm import tqdm

def board_to_vector (board):
    vector = np.zeros((12, 64), dtype=np.float32)
    for piece_type in range(1, 7):
        for color in [chess.WHITE, chess.BLACK]:
            idx = (piece_type - 1) if color == chess.WHITE else (piece_type + 5)
            bitboard = board.pieces(piece_type, color)
            for square in bitboard:
                vector[idx][square] = 1.0
    return vector.flatten()

def process_pgn (pgn_path, output_dir, max_games=1000):
    X, y = [], []
    if not os.path.exists(pgn_path):
        print(f"Error: No PGN file found at {pgn_path}")
        return
    
    with open(pgn_path) as pgn:
        for _ in tqdm(range(max_games), desc="Parsing Games"):
            game = chess.pgn.read_game(pgn)
            if game is None: break

            board = game.board()

            for move in game.mainline_moves():
                X.append(board_to_vector(board))
                y.append(move.from_square * 64 + move.to_square)
                board.push(move)

    np.save(os.path.join(output_dir, "X_train.npy"), np.array(X))
    np.save(os.path.join(output_dir, "y_train.npy"), np.array(y))
    print(f"\nSaved {len(X)} positions to {output_dir}")

if __name__ == "__main__":
    process_pgn("python/data/games.pgn", "python/data/")