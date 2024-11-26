# Import necessary libraries
import numpy as np
import chess
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
import sys

# Function to convert FEN to tensor
def fen_to_tensor(fen):
    piece_to_index = {
        'P': 0,  'N': 1,  'B': 2,  'R': 3,  'Q': 4,  'K': 5,
        'p': 6,  'n': 7,  'b': 8,  'r': 9,  'q': 10, 'k': 11
    }
    board_tensor = np.zeros((8, 8, 12), dtype=int)
    fen_board = fen.split(' ')[0]
    rows = fen_board.split('/')
    for row_idx, row in enumerate(rows):
        col_idx = 0
        for char in row:
            if char.isdigit():
                col_idx += int(char)
            else:
                piece_idx = piece_to_index.get(char)
                if piece_idx is not None:
                    board_tensor[row_idx, col_idx, piece_idx] = 1
                else:
                    print(f"Invalid character '{char}' in FEN.")
                    sys.exit(1)
                col_idx += 1
    return board_tensor

# Helper functions to calculate parameters
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
    original_turn = board.turn
    board.turn = chess.WHITE
    white_moves = len(list(board.legal_moves))
    board.turn = chess.BLACK
    black_moves = len(list(board.legal_moves))
    board.turn = original_turn  # Restore the original turn
    return white_moves - black_moves

def calculate_king_safety(board):
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

def get_game_phase(turn_number):
    if turn_number <= 15:
        return {'opening': 1, 'middle_game': 0, 'endgame': 0}
    elif turn_number <= 40:
        return {'opening': 0, 'middle_game': 1, 'endgame': 0}
    else:
        return {'opening': 0, 'middle_game': 0, 'endgame': 1}

def calculate_pawn_structure(board):
    def pawn_structure_for_color(color):
        pawns = board.pieces(chess.PAWN, color)
        files_with_pawns = [chess.square_file(square) for square in pawns]
        file_counts = {}
        for file in files_with_pawns:
            file_counts[file] = file_counts.get(file, 0) + 1

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
    key_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
    white_control = sum(1 for square in key_squares if board.is_attacked_by(chess.WHITE, square))
    black_control = sum(1 for square in key_squares if board.is_attacked_by(chess.BLACK, square))
    return white_control - black_control

# Function to calculate all parameters
def calculate_parameters(board, turn_number, elo_diff=0):
    piece_diff = calculate_piece_differential(board)
    mobility = calculate_mobility(board)
    king_safety = calculate_king_safety(board)
    control_of_key_squares = calculate_control_of_key_squares(board)
    game_phase = get_game_phase(turn_number)
    pawn_structure = calculate_pawn_structure(board)

    params = {
        'turn_number': turn_number,
        'elo_diff': elo_diff,
        'piece_diff': piece_diff,
        'mobility': mobility,
        'king_safety': king_safety,
        'control_of_key_squares': control_of_key_squares,
        'doubled_pawns_diff': pawn_structure['doubled_pawns_diff'],
        'isolated_pawns_diff': pawn_structure['isolated_pawns_diff'],
        'opening': game_phase['opening'],
        'middle_game': game_phase['middle_game'],
        'endgame': game_phase['endgame']
    }
    return params

# Function to ensure all probability are between 0 and 1.
def warp_probability(predicted_y, k=10):
    """
    Warp the predicted probability to be between (0, 1) exclusive, while preserving
    values near 0.5 and warping values closer to 0 and 1.
    
    Parameters:
        predicted_y (float): The raw predicted probability.
        k (float): The steepness of the warp effect. Default is 10.
    
    Returns:
        float: Warped probability.
    """
    raw_prob = predicted_y[0][0]
    warped_prob = 1 / (1 + np.exp(-k * (raw_prob - 0.5)))
    return warped_prob

def main():
    # Load the best model
    model_path = 'saved_models/model_1/best_model.h5'  
    try:
        model = load_model(model_path, compile=False)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Prompt the user for FEN notation
    fen = input("\nEnter the FEN notation of the board state: ").strip()
    try:
        board = chess.Board(fen)
    except ValueError as e:
        print(f"Invalid FEN notation: {e}")
        sys.exit(1)

    # Check for the initial position
    if board.board_fen() == chess.STARTING_BOARD_FEN:
        print("\nPredicted Win Probability for White: 50.00%")
        sys.exit(0)

    # Prompt the user for move count (turn number), default to 55 if not provided
    turn_number_input = input("Enter the move count (turn number) [default: 55]: ").strip()
    if turn_number_input == '':
        turn_number = 55
    else:
        try:
            turn_number = int(turn_number_input)
        except ValueError:
            print("Invalid move count. Please enter an integer.")
            sys.exit(1)

    # Prompt the user for Elo difference, default to 0 if not provided
    elo_diff_input = input("Enter the Elo difference (White Elo - Black Elo) [default: 0]: ").strip()
    if elo_diff_input == '':
        elo_diff = 0
    else:
        try:
            elo_diff = float(elo_diff_input)
        except ValueError:
            print("Invalid Elo difference. Please enter a number.")
            sys.exit(1)

    # Convert FEN to tensor
    board_tensor = fen_to_tensor(fen)
    X_board_input = np.expand_dims(board_tensor, axis=0)  # Add batch dimension

    # Calculate parameters
    params = calculate_parameters(board, turn_number, elo_diff)

    # Prepare parameters for model input
    parameter_columns = [
        'turn_number', 'elo_diff', 'piece_diff', 'mobility', 'king_safety',
        'control_of_key_squares', 'opening', 'middle_game', 'endgame',
        'doubled_pawns_diff', 'isolated_pawns_diff'
    ]
    params_list = [params[col] for col in parameter_columns]
    X_params_input = np.array([params_list], dtype=np.float32)

    # Predict the win probability
    predicted_y = model.predict([X_board_input, X_params_input])
    predicted_win_probability = predicted_y[0][0]
    predicted_win_probability = warp_probability(predicted_y)

    print(f"\nPredicted Win Probability for White: {predicted_win_probability * 100:.2f}%")

if __name__ == "__main__":
    main()