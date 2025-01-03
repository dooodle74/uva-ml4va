from flask import Flask, request, jsonify
from flask_cors import CORS
import chess
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

import chess_model_predictor


app = Flask(__name__)
CORS(app)  # Optional: if you want to allow cross-origin calls
model = load_model("model/best_model.h5")  # load your .h5 model once at startup

@app.route("/", methods=["GET"])
def index():
    return "Chess Predictor API is up!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    fen = data.get("fen", None)

    if not fen:
        return jsonify({"error": "No FEN provided"}), 400

    # Use your predict_from_fen logic
    board = chess.Board(fen)
    piece_diff = chess_model_predictor.calculate_piece_differential(board)
    mobility = chess_model_predictor.calculate_mobility(board)
    king_safety = chess_model_predictor.calculate_king_safety(board)
    control_key = chess_model_predictor.calculate_control_of_key_squares(board)
    pawn_struct = chess_model_predictor.calculate_pawn_structure(board)
    doubled_pawns_diff = pawn_struct['doubled_pawns_diff']
    isolated_pawns_diff = pawn_struct['isolated_pawns_diff']
    
    board_tensor = chess_model_predictor.fen_to_tensor(fen)
    extra_features = np.array([
        piece_diff, mobility, king_safety,
        control_key, doubled_pawns_diff, isolated_pawns_diff
    ], dtype=np.float32)

    board_input = np.expand_dims(board_tensor, axis=0)
    extra_input = np.expand_dims(extra_features, axis=0)

    probability = model.predict([board_input, extra_input])[0, 0]

    return jsonify({
        "probability": float(probability)
    })