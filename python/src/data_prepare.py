import chess.pgn
import numpy as np
import os
from tqdm import tqdm

def board_to_tensor(board):
    # 12 layers: 6 for White, 6 for Black
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    piece_map = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
                 "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11}
    
    for square, piece in board.piece_map().items():
        rank, file = divmod(square, 8)
        tensor[piece_map[piece.symbol()], rank, file] = 1.0
    return tensor # Now returns 12x8x8

def process_huge_pgn(pgn_path, output_dir, target_positions=1000000):
    X, y, z = [], [], []
    count = 0
    with open(pgn_path, encoding="utf-8", errors="replace") as pgn:
        pbar = tqdm(total=target_positions, desc="Parsing Master Games")
        while count < target_positions:
            game = chess.pgn.read_game(pgn)
            if game is None: break
            
            # ELO Filter
            white_elo = game.headers.get("WhiteElo", "0")
            black_elo = game.headers.get("BlackElo", "0")
            try:
                if int(white_elo) < 2200 or int(black_elo) < 2200: continue
            except: continue

            res = game.headers.get("Result", "*")
            reward = 1.0 if res == "1-0" else (-1.0 if res == "0-1" else 0.0)
            
            board = game.board()
            for move in game.mainline_moves():
                # Flatten only for saving to disk
                X.append(board_to_tensor(board).flatten())
                y.append(move.from_square * 64 + move.to_square)
                z.append(reward if board.turn == chess.WHITE else -reward)
                board.push(move)
                count += 1
                pbar.update(1)
                if count >= target_positions: break
    
    np.save(os.path.join(output_dir, "X_train.npy"), np.array(X))
    np.save(os.path.join(output_dir, "y_train.npy"), np.array(y))
    np.save(os.path.join(output_dir, "z_train.npy"), np.array(z))

if __name__ == "__main__":
    process_huge_pgn("python/data/games.pgn", "python/data/")