# Chess Result Prediction

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

**Mobility**

Difference in number of legal moves. 

**King Safety**

Points are added for certain pawn sheild structures around the king, and are subtracted if less of these pawns exist. Return the difference between players.

**Pawn Structure**

Two types of pawn structures are evaluated:
- Isolated pawns
- Doubled pawns
Differential is calculated between white and black, and returned. 

**Control of Key Squares**

Bonus points for controlling crucial middle section squares.

## Model Training

We used a primary CNN architecture with the parameters mentioned above as additional features. The tensors we used consisted of:
-	Each chess board: 8x8x13 tensor. 8x8 represents the board squares and 13 represents the 6 possible pieces from each side, plus one for an unoccupied square.
-	Features: 6-D vector, representing the six additional features mentioned above.
-	Ground truth: separate vector for Stockfish generated ground truth data, which obviously only used in the training set. 
The board input undergoes two convolutional blocks with 64 and 128 filters, respectively, using ReLU activations, batch normalization, and pooling to extract spatial features. A global average pooling layer reduces spatial dimensions, followed by dense layers for further processing. The extra features input is processed through fully connected layers with ReLU activations and batch normalization. Both branches are merged via concatenation, followed by additional dense layers for joint feature learning, with dropout layers for regularization. The final output is a single sigmoid-activated node for predicting a value in the range [0,1], optimized using the Adam optimizer and mean squared error loss, with mean absolute error tracked as a metric. In our case, the mean absolute error directly reflects the difference in prediction of white win between our own model and the ground truth Stockfish evaluation. 

### Using the Model
The final code block of the `chess_model_training.ipynb` Jupyter notebook can be run and output our prediction. Make sure certain blocks beforehand are ran so the relevant methods exist in memory.

In addition, running python `chess_model_predictor.py` with first positional argument fen state (surround in quotes) and optional `-m` flag for the model path, can achieve the same result. 

### Results
CSV file and image plot of the latest training is located in `final_model`
