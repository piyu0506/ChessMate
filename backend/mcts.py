import math
import torch
import numpy as np
import chess
from encoder import encode_move

class MCTSNode:
    def __init__(self, board, parent=None, prior=0.0):
        self.board = board
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.is_expanded = False

    def value(self):
        if self.visit_count == 0: return 0.0
        return self.value_sum / self.visit_count

class MCTS:
    def __init__(self, model, device, simulations=600):
        self.model = model
        self.device = device
        self.simulations = simulations
        self.c_puct = 0.8

    def search(self, board):
        root = MCTSNode(board)
        self.expand(root)
        
        for _ in range(self.simulations):
            node = root
            # 1. SELECT
            while node.is_expanded and not node.board.is_game_over():
                move = self.select_child(node)
                if move is None or move not in node.children:
                    break
                node = node.children[move]
            
            # 2. EXPAND & EVALUATE
            value = 0.0
            if not node.board.is_game_over():
                if not node.is_expanded:
                    value = self.expand(node)
                else:
                    value = node.value()
            else:
                outcome = node.board.outcome()
                if outcome is None:
                    value = 0.0
                elif outcome.winner is None:
                    value = 0.0  # stalemate/draw
                else:
                    value = -1.0  # checkmate, loser is current node's turn
            
            # 3. BACKPROP
            self.backpropagate(node, value)
            
        # Select best move (Most Visited)
        best_move = None
        max_visits = -1
        for move, child in root.children.items():
            if child.visit_count > max_visits:
                max_visits = child.visit_count
                best_move = move
        return best_move

    def select_child(self, node):
        best_score = -float('inf')
        best_move = None
        for move, child in node.children.items():
            # Standard AlphaZero PUCT
            q_val = -child.value()
            u_val = self.c_puct * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)
            score = q_val + u_val
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def expand(self, node):
        state = self.board_to_tensor(node.board).unsqueeze(0).to(self.device)
        with torch.no_grad():
            p_logits, v_val = self.model(state)

        # MASKING ILLEGAL MOVES — stable softmax over legal moves only
        p_logits = p_logits[0].cpu().numpy()
        legal_moves = list(node.board.legal_moves)

        valid_indices = []
        valid_moves = []
        for move in legal_moves:
            idx = encode_move(move, node.board.turn)
            if idx is not None:
                valid_indices.append(idx)
                valid_moves.append(move)

        if valid_moves:
            logits = p_logits[valid_indices]
            logits = logits - logits.max()  # numerical stability
            probs = np.exp(logits)
            probs /= probs.sum()
        else:
            # fallback: uniform over all legal moves
            valid_moves = legal_moves
            probs = np.full(len(legal_moves), 1.0 / max(len(legal_moves), 1))

        for move, prior in zip(valid_moves, probs):
            child_board = node.board.copy()
            child_board.push(move)
            node.children[move] = MCTSNode(child_board, parent=node, prior=float(prior))

        node.is_expanded = True
        return v_val.item()

    def backpropagate(self, node, value):
        curr = node
        curr_val = value
        while curr is not None:
            curr.visit_count += 1
            curr.value_sum += curr_val
            curr_val = -curr_val
            curr = curr.parent

    def board_to_tensor(self, board):
        tensor = np.zeros((12, 8, 8), dtype=np.float32)
        piece_map = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
                     "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11}
        for square, piece in board.piece_map().items():
            rank, file = divmod(square, 8)
            tensor[piece_map[piece.symbol()], rank, file] = 1.0
        return torch.from_numpy(tensor).float()