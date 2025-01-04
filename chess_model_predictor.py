import chess
import chess.pgn
import pandas as pd
import re
from collections import Counter
from tqdm import tqdm
import numpy as np
from keras.models import load_model
import sys

def calculate_piece_differential(board):
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0 
    }
    white_value = sum(
        piece_values.get(piece.piece_type, 0)
        for piece in board.piece_map().values()
        if piece.color == chess.WHITE
    )
    black_value = sum(
        piece_values.get(piece.piece_type, 0)
        for piece in board.piece_map().values()
        if piece.color == chess.BLACK
    )
    return white_value - black_value

def calculate_mobility(board):
    """
    Calculate the mobility difference (white_moves - black_moves).

    Args:
        board (chess.Board): The current game state.

    Returns:
        int: The difference in the number of legal moves.
    """
    original_turn = board.turn
    board.turn = chess.WHITE
    white_moves = len(list(board.legal_moves))
    board.turn = chess.BLACK
    black_moves = len(list(board.legal_moves))
    board.turn = original_turn  # Restore the original turn
    return white_moves - black_moves

def calculate_king_safety(board):
    """
    Calculate the king safety differential.

    Args:
        board (chess.Board): The current game state.

    Returns:
        float: The difference in king safety scores (white - black).
    """
    def king_safety_for_color(color):
        king_square = board.king(color)
        safety_score = 0

        # Check if the king is castled
        if color == chess.WHITE:
            if king_square == chess.E1:
                safety_score -= 1  # Less safe if not castled
            elif king_square in [chess.G1, chess.C1]:
                safety_score += 1  # More safe if castled
        else:
            if king_square == chess.E8:
                safety_score -= 1
            elif king_square in [chess.G8, chess.C8]:
                safety_score += 1

        # Check for pawn shield around the king
        pawn_shield_squares = []
        if color == chess.WHITE:
            if king_square == chess.G1:
                pawn_shield_squares = [chess.F2, chess.G2, chess.H2]
            elif king_square == chess.C1:
                pawn_shield_squares = [chess.A2, chess.B2, chess.C2]
            else:
                pawn_shield_squares = [chess.D2, chess.E2, chess.F2]
        else:
            if king_square == chess.G8:
                pawn_shield_squares = [chess.F7, chess.G7, chess.H7]
            elif king_square == chess.C8:
                pawn_shield_squares = [chess.A7, chess.B7, chess.C7]
            else:
                pawn_shield_squares = [chess.D7, chess.E7, chess.F7]

        for square in pawn_shield_squares:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                safety_score += 0.5
            else:
                safety_score -= 0.5

        return safety_score

    white_king_safety = king_safety_for_color(chess.WHITE)
    black_king_safety = king_safety_for_color(chess.BLACK)
    return white_king_safety - black_king_safety

def calculate_pawn_structure(board):
    """
    Calculate pawn structure metrics like doubled and isolated pawns.

    Args:
        board (chess.Board): The current game state.

    Returns:
        dict: Differences in pawn structure metrics (white - black).
    """
    def pawn_structure_for_color(color):
        pawns = board.pieces(chess.PAWN, color)
        files_with_pawns = [chess.square_file(square) for square in pawns]
        file_counts = Counter(files_with_pawns)

        # Doubled pawns: files with more than one pawn
        doubled_pawns = sum(1 for count in file_counts.values() if count > 1)

        # Isolated pawns: pawns with no friendly pawns on adjacent files
        isolated_pawns = 0
        for file in file_counts:
            adjacent_files = [file - 1, file + 1]
            has_adjacent_pawn = any(
                adj_file in file_counts for adj_file in adjacent_files if 0 <= adj_file <= 7
            )
            if not has_adjacent_pawn:
                isolated_pawns += file_counts[file]

        return {'doubled_pawns': doubled_pawns, 'isolated_pawns': isolated_pawns}

    white_pawn_structure = pawn_structure_for_color(chess.WHITE)
    black_pawn_structure = pawn_structure_for_color(chess.BLACK)

    pawn_structure_diff = {
        'doubled_pawns_diff': white_pawn_structure['doubled_pawns'] - black_pawn_structure['doubled_pawns'],
        'isolated_pawns_diff': white_pawn_structure['isolated_pawns'] - black_pawn_structure['isolated_pawns']
    }
    return pawn_structure_diff

def calculate_control_of_key_squares(board):
    """
    Calculate the difference in control of key squares.

    Args:
        board (chess.Board): The current game state.

    Returns:
        int: Difference in control of key squares (white - black).
    """
    key_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
    white_control = sum(1 for square in key_squares if board.is_attacked_by(chess.WHITE, square))
    black_control = sum(1 for square in key_squares if board.is_attacked_by(chess.BLACK, square))
    return white_control - black_control

def fen_to_tensor(fen):
    """
    Convert a FEN string into a 8x8x13 numpy array.
    If you wish to use symmetrical indexing, adjust piece_to_channel accordingly.
    """
    piece_to_channel = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    
    board = chess.Board(fen)
    board_tensor = np.zeros((8, 8, 13), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        rank = 7 - (square // 8)
        file = square % 8
        if piece:
            ch = piece_to_channel[piece.symbol()]
            board_tensor[rank, file, ch] = 1.0
        else:
            board_tensor[rank, file, 12] = 1.0

    return board_tensor

# tf.get_logger().setLevel('ERROR')

def predict_from_fen(fen, model_path="best_model.h5"):
    # Load the model
    model = load_model(model_path)
    
    # Convert fen to board tensor
    board_tensor = fen_to_tensor(fen)
    
    # Compute extra features
    board = chess.Board(fen)
    piece_diff = calculate_piece_differential(board)
    mobility = calculate_mobility(board)
    king_safety = calculate_king_safety(board)
    control_key = calculate_control_of_key_squares(board)
    pawn_struct = calculate_pawn_structure(board)
    doubled_pawns_diff = pawn_struct['doubled_pawns_diff']
    isolated_pawns_diff = pawn_struct['isolated_pawns_diff']
    
    # Combine
    extra_features = np.array([piece_diff, mobility, king_safety, control_key, doubled_pawns_diff, isolated_pawns_diff], dtype=np.float32)
    
    # Reshape for prediction (add batch dimension)
    board_input = np.expand_dims(board_tensor, axis=0)   # Shape: (1, 8, 8, 13)
    extra_input = np.expand_dims(extra_features, axis=0) # Shape: (1, 6)
    
    # Predict
    prediction = model.predict([board_input, extra_input])
    return prediction[0, 0]

def main():
    if len(sys.argv) < 2:
        print("Usage: python chess_model_predictor.py <input_fen> [-m <optional_model_path>]")
        sys.exit(1)

    # Default values
    user_input = None
    flag_m_value = "best_model.h5"

    args = iter(sys.argv[1:])
    for arg in args:
        if arg == "-m":
            try:
                next_arg = next(args)
                if next_arg.startswith("-"): 
                    flag_m_value = None
                else:
                    flag_m_value = next_arg
            except StopIteration:
                flag_m_value = None
        else:
            user_input = arg

    if not user_input:
        print("Error: FEN input is required.")
        sys.exit(1)

    fen = user_input

    if flag_m_value:
        model_path = flag_m_value
    else:
        model_path = "best_model.h5"

    pred = predict_from_fen(fen, model_path = model_path)
    print()
    print(f"Input state: {fen}")
    print("Predicted probability of White win:", pred)
    print()
    board = chess.Board(fen)
    chess.svg.board(board, size=350)

if __name__ == "__main__":
    main()