import torch
import numpy as np
import chess
from model import ChessNet

def predict_best_move(board_fen, model_path="python/models/chess_model.pth"):
    # 1. Setup Board and Model
    board = chess.Board(board_fen)
    model = ChessNet()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # 2. Vectorize the board (Matching your C++ logic)
    def board_to_tensor(board):
        vector = np.zeros((1, 12, 8, 8), dtype=np.float32)
        for piece_type in range(1, 7):
            for color in [chess.WHITE, chess.BLACK]:
                idx = (piece_type - 1) if color == chess.WHITE else (piece_type + 5)
                for square in board.pieces(piece_type, color):
                    vector[0][idx][square // 8][square % 8] = 1.0
        return torch.from_numpy(vector)

    # 3. Predict
    with torch.no_grad():
        policy, value = model(board_to_tensor(board))
        probabilities = torch.softmax(policy, dim=1).flatten()

    # 4. Filter for only legal moves
    legal_moves = list(board.legal_moves)
    move_scores = []
    for move in legal_moves:
        idx = move.from_square * 64 + move.to_square
        move_scores.append((move, probabilities[idx].item()))

    # Sort by score
    move_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nPosition Evaluation: {value.item():.4f}")
    print("Top 3 Suggested Moves:")
    for i in range(min(3, len(move_scores))):
        move, score = move_scores[i]
        print(f"{i+1}. {move} (Confidence: {score*100:.2f}%)")

if __name__ == "__main__":
    # Test with Starting Position
    predict_best_move(chess.STARTING_FEN)