# Chess Result Prediction
Training loop stored in `chess_model_training.ipynb`

To use the model, use `predict_win_probability.py` Python file.

Upon running, you are asked to input the board state given in standard FEN format. A generator can be seen here: [https://www.redhotpawn.com/chess/chess-fen-viewer.php].

You are also asked to provide a move number for this move.

The program will output the prediction for white win.

## Current Issues
- Overfitting
- Issue with loading the saved model (.keras format is a headache)

The above is being looked at in a new training session, and the training results should be done soon. - David Wang 11/25 4:41 am