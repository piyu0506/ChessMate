import chess
import numpy as np

# --- 73-PLANE SPATIAL ENCODING ---
# Maps moves to geometry (Direction + Distance) rather than flat numbers.

def encode_move(move: chess.Move, turn: bool) -> int:
    """
    Maps a chess.Move to an index [0, 4671].
    Formula: (From_Square * 73) + Plane_Index
    """
    f_rank, f_file = divmod(move.from_square, 8)
    t_rank, t_file = divmod(move.to_square, 8)
    delta_rank = t_rank - f_rank
    delta_file = t_file - f_file

    # 1. UNDERPROMOTIONS (Planes 64-72)
    # Only for Knight, Bishop, Rook. Queen promo is treated as a normal move.
    if move.promotion and move.promotion in [chess.KNIGHT, chess.BISHOP, chess.ROOK]:
        p_idx = 0 # Knight
        if move.promotion == chess.BISHOP: p_idx = 1
        elif move.promotion == chess.ROOK: p_idx = 2
        
        # Direction: -1 (Left), 0 (Forward), 1 (Right) relative to White
        # We simplify to absolute file diff for this implementation
        direction = 1 # Center
        if delta_file == -1: direction = 0 
        elif delta_file == 1: direction = 2 
        
        plane = 64 + (direction * 3) + p_idx
        return (move.from_square * 73) + plane

    # 2. KNIGHT MOVES (Planes 56-63)
    knight_jumps = [
        (2, 1), (1, 2), (-1, 2), (-2, 1), 
        (-2, -1), (-1, -2), (1, -2), (2, -1)
    ]
    if (delta_rank, delta_file) in knight_jumps:
        plane = 56 + knight_jumps.index((delta_rank, delta_file))
        return (move.from_square * 73) + plane

    # 3. QUEEN MOVES (Planes 0-55)
    # N, NE, E, SE, S, SW, W, NW
    dirs = [
        (1, 0), (1, 1), (0, 1), (-1, 1), 
        (-1, 0), (-1, -1), (0, -1), (1, -1)
    ]
    
    for d_idx, (dr, df) in enumerate(dirs):
        for dist in range(1, 8):
            if delta_rank == dr * dist and delta_file == df * dist:
                plane = (d_idx * 7) + (dist - 1)
                return (move.from_square * 73) + plane

    return None # Should not happen for legal moves

def decode_move(idx: int, board: chess.Board) -> chess.Move:
    # Necessary for MCTS to interpret the Network's output
    from_sq, plane = divmod(idx, 73)
    f_rank, f_file = divmod(from_sq, 8)
    
    delta_rank, delta_file = 0, 0
    promotion = None
    
    # A. Queen Moves
    if 0 <= plane < 56:
        d_idx, dist = divmod(plane, 7)
        dist += 1
        dirs = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
        dr, df = dirs[d_idx]
        delta_rank, delta_file = dr * dist, df * dist
        
    # B. Knight Moves
    elif 56 <= plane < 64:
        knight_jumps = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
        delta_rank, delta_file = knight_jumps[plane - 56]
        
    # C. Underpromotions
    elif 64 <= plane < 73:
        residue = plane - 64
        direction = residue // 3
        promotion = [chess.KNIGHT, chess.BISHOP, chess.ROOK][residue % 3]
        
        # Simple absolute direction logic
        is_white = (f_rank == 6) # Promoting from rank 6
        forward = 1 if is_white else -1
        
        delta_rank = forward
        if direction == 0: delta_file = -1
        elif direction == 1: delta_file = 0
        elif direction == 2: delta_file = 1

    t_rank = f_rank + delta_rank
    t_file = f_file + delta_file
    
    if not (0 <= t_rank <= 7 and 0 <= t_file <= 7): return None
        
    to_sq = chess.square(t_file, t_rank)
    
    # Auto-detect Queen Promotion
    move = chess.Move(from_sq, to_sq, promotion)
    if promotion is None:
        p = board.piece_at(from_sq)
        if p and p.piece_type == chess.PAWN:
            if (t_rank == 7 and p.color == chess.WHITE) or (t_rank == 0 and p.color == chess.BLACK):
                move = chess.Move(from_sq, to_sq, chess.QUEEN)
                
    return move