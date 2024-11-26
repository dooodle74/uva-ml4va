# Chess Result Prediction
Training loop stored in `chess_model_training.ipynb`

To use the model, scroll down to part 5 and manually enter the parameters.

Alternatively, use `predict_win_probability.py` Python file for a more legacy version.

Upon running, you are asked to input the board state given in standard FEN format. A generator can be seen here: [https://www.redhotpawn.com/chess/chess-fen-viewer.php].

You are also asked to provide a move number for this move.

The program will output the prediction for white win.

# Project Parts and Notes (Also found in Notebook)

## Game State Construction and Data Engineering

We want to make a prediction of a winner (white win percentage) based on a particular board state. To extract the board states, we extract every single board state from each game, using the sequences of moves.

We then calculate certain parameters from the game state and save all individual game states. The details are as follows. All differential values are (WHITE - BLACK).

**Piece Differential**

Each remaining piece gets assigned a standard weight:
- PAWN: 1
- KNIGHT: 3
- BISHOP: 3
- ROOK: 5
- QUEEN: 9
- KING: 0 (as all states must have a king)
The differntial is calculated by white's piece value minus that of black.

**ELO Differential**

Difference of ELO rating.

**Mobility**

Difference in number of legal moves. 

**King Safety**

Points are added for certain pawn sheild structures around the king, and are subtracted if less of these pawns exist. Return the difference between players.

**Game Phase**

Return the game phase (opening, midgame, endgame) based on move number and remaining pieces. 

**Pawn Structure**

Two types of pawn structures are evaluated:
- Isolated pawns
- Doubled pawns
Differential is calculated between white and black, and returned. 

**Control of Key Squares**

Bonus points for controlling crucial middle section squares.

## Model Training

We use our own CNN model to train the data. We first transform the board into a tensor, then normalize the above parameters to enhance the CNN structure. 

### Output Data
One of the main problems with our idea is that we don't have a definitive prediction of a game's win probability status for every single game state - we are only given the eventual winner for training. Thus, after research and experimenting, we devised a formula, mostly linear, based on the final win party and game state, to assign a win probability to each state. Our model is now aimed to produce this prediction based on the above parameters with no knowledge of the winner.