import os
import torch
import chess

from model import ChessNet
from mcts import MCTS


class ChessEngine:
    def __init__(self):
        torch.set_num_threads(1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_path = os.path.join(
            os.path.dirname(__file__),
            "models",
            "chess_model.pth"
        )

        self.model = ChessNet(num_res_blocks=6).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()

        self.mcts = MCTS(
            self.model,
            self.device,
            simulations=40
        )

    def get_ai_response(self, fen: str, player_move: str):
        board = chess.Board(fen)

        move = chess.Move.from_uci(player_move)

        if move not in board.legal_moves:
            return {
                "success": False,
                "message": "Illegal move."
            }

        board.push(move)

        if board.is_game_over():
            return {
                "success": True,
                "ai_move": None,
                "fen": board.fen(),
                "game_over": True,
                "result": board.result()
            }

        with torch.inference_mode():
            ai_move = self.mcts.search(board)

        if ai_move is None:
            return {
                "success": False,
                "message": "AI failed to generate a move."
            }

        board.push(ai_move)

        return {
            "success": True,
            "ai_move": ai_move.uci(),
            "fen": board.fen(),
            "game_over": board.is_game_over(),
            "result": board.result() if board.is_game_over() else None
        }