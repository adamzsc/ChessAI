import chess
import lichess # This is not standard, assuming it's a wrapper for requests/lichess API calls
import joblib
import torch
import time

import numpy as np

# Assuming board_utils contains necessary helper functions like load_model,
# get_material_imbalance, get_board_channels, and the CNN_Regressor definition.
import board_utils as b 
import requests

def user_go(board):
    """Handles user input for a move."""
    while True:
        try:
            move_input = input("Enter Move (e.g., e2e4): ")
            move = chess.Move.from_uci(move_input)
            if move in board.legal_moves:
                return move
            else:
                print("NOT A LEGAL MOVE. Try again.")
        except ValueError:
            print("Invalid move format. Use standard UCI (e.g., e2e4).")

class Engine():
    
    def __init__(self,engine_path,scalers_board_filename,scaler_extra_filename,scaler_y_filename):
        
        # Load pre-trained scalers and the PyTorch model
        self.scalers_board = joblib.load("scalers/"+scalers_board_filename)
        self.scaler_extra = joblib.load("scalers/"+scaler_extra_filename)
        self.scaler_y = joblib.load("scalers/"+scaler_y_filename)
        
        self.n_input_channels = 20
        self.extra_features_size = 1
        # Assumes b.load_model correctly loads the PyTorch model from the .pth file
        self.model = b.load_model(b.CNN_Regressor, engine_path, self.n_input_channels, self.extra_features_size)
        
        self.transposition_table = {}
        self.killer_moves = [[None, None] for _ in range(100)]    
        self.principal_variation_move = None 
        
        # Use a slightly smaller mate score for better margin/detection
        self.MATE_SCORE = 1000000 
        
        # MVV_LVA Table (Most Valuable Victim, Least Valuable Attacker)
        self.MVV_LVA =  np.array([   #Accessed [victim_type - 1][attacker_type - 1]
            [15, 14, 13, 12, 11, 10], # victim P, attacker P, N, B, R, Q, K
            [25, 24, 23, 22, 21, 20], # victim N, attacker P, N, B, R, Q, K   
            [35, 34, 33, 32, 31, 30], # victim B, attacker P, N, B, R, Q, K
            [45 ,44, 43, 42, 41, 40], # victim R, attacker P, N, B, R, Q, K
            [55, 54, 53, 52, 51, 50], # victim Q, attacker P, N, B, R, Q, K
            ] )    
        
    def find_best_move(self, board, time_limit_seconds=10.0):             
        
        # --- Opening Book Check ---
        if board.fullmove_number < 5:
            fen = board.fen()
            opening_data = self.get_lichess_opening_data(fen)
            if opening_data and 'moves' in opening_data and len(opening_data['moves']) > 0:
                n_moves = len(opening_data['moves'])
                move_choice = np.random.randint(0,n_moves)
                move = opening_data['moves'][move_choice]['uci']
                try:
                    return chess.Move.from_uci(move)
                except ValueError:
                    pass
            
        start_time = time.time()
        current_best_move = None
        predicted_score = 0.0  # Predicted score from the previous depth search
        
        # Initial aspiration window size (50 centipawns or 0.5 score units)
        ASPIRATION_WINDOW_DELTA = 0.5 
        
        # --- Iterative Deepening Loop ---
        for depth in range(1,100):
            
            if (time.time() - start_time) >= time_limit_seconds:
                break
            
            # --- Setup for Aspiration Window Search for the current depth ---
            
            # Use the initial delta for the first search pass at this depth
            delta = ASPIRATION_WINDOW_DELTA 
            
            # Reset the best move/score for this depth
            best_score_this_depth = -np.inf
            best_move_this_depth = None
            
            # 1. Root Move Ordering (done only once per depth)
            moves = list(board.legal_moves)        
            scored_moves = []
            for move in moves:
                move_score = self.score_move(board, move, depth) 
                scored_moves.append((move_score, move))
            scored_moves.sort(key=lambda x: x[0], reverse=True)

            
            # --- Aspiration Window Loop (Retries the search if the score is outside the window) ---
            while True:
                
                # 2. Set the Window Bounds for the current search pass
                if depth == 1 or abs(predicted_score) > self.MATE_SCORE - 1000:
                    # For shallow depth or near-mate positions, use the full window
                    alpha_bound = -self.MATE_SCORE 
                    beta_bound = self.MATE_SCORE
                else:
                    # Set narrow window around the predicted score, clipped by mate bounds
                    alpha_bound = max(-self.MATE_SCORE, predicted_score - delta)
                    beta_bound = min(self.MATE_SCORE, predicted_score + delta)
                
                # Reset tracking variables for this specific search pass
                score_in_this_pass = -np.inf
                move_in_this_pass = None
                
                # The alpha used in the loop starts at the lower bound of the window
                search_alpha = alpha_bound 
                
                # 3. Iterate through sorted moves
                for move_score, move in scored_moves:                
                    
                    if (time.time() - start_time) >= time_limit_seconds:
                        return current_best_move 
                    
                    # CRITICAL EFFICIENCY FIX: Use push/pop
                    board.push(move)
                
                    # Pass the search bounds to negamax (note the inversion for the maximizing player)
                    score = -self.negamax(board, depth - 1, -beta_bound, -search_alpha, -1) 
                        
                    board.pop()
                    
                    if score > score_in_this_pass:
                        score_in_this_pass = score
                        move_in_this_pass = move
                        
                    search_alpha = max(search_alpha, score_in_this_pass) # Update the root alpha value
                    
                    # Alpha-Beta Cutoff within the root search (Fail-High)
                    if search_alpha >= beta_bound:
                        break # Fail-High, we broke the upper bound

                # --- Aspiration Window Failure Check ---
                
                # If we used the full window, the search is complete for this depth
                if alpha_bound == -self.MATE_SCORE and beta_bound == self.MATE_SCORE:
                    best_score_this_depth = score_in_this_pass
                    best_move_this_depth = move_in_this_pass
                    break # Full search finished successfully
                    
                # Success: Score landed within the initial narrow window (alpha_bound < score < beta_bound)
                if score_in_this_pass > alpha_bound and score_in_this_pass < beta_bound:
                    best_score_this_depth = score_in_this_pass
                    best_move_this_depth = move_in_this_pass
                    break
                
                # Fail-Low: Score is <= alpha_bound. Re-search with a wider window on the low side.
                elif score_in_this_pass <= alpha_bound:
                    # Set the best score found so far as the new prediction point
                    predicted_score = score_in_this_pass
                    # Double the delta for the next search pass
                    delta *= 2 
                    continue
                
                # Fail-High: Score is >= beta_bound. Re-search with a wider window on the high side.
                elif score_in_this_pass >= beta_bound:
                    # Set the best score found so far as the new prediction point
                    predicted_score = score_in_this_pass
                    # Double the delta for the next search pass
                    delta *= 2
                    continue


            # --- Depth Finished (Success) ---
            if best_move_this_depth is not None:
                current_best_move = best_move_this_depth
                # Use the accurate score found in the last search pass as the prediction for the next depth
                predicted_score = best_score_this_depth 
                
                # Store the PV move for Move Ordering in the next iteration
                self.principal_variation_move = current_best_move 
                
                # Optional: Print results after each full depth
                print(f"Depth {depth} completed. Best Move: {current_best_move} | Score: {predicted_score:.2f}")


        # Return the best move found in the last full iteration
        return current_best_move
        
    def score_move(self,board,move,current_depth):
        
        score = 100
        
        if move == self.principal_variation_move:
            return 1000000 # Highest score
        
        # --- Killer Moves ---
        # Note: Killer moves are only relevant in the main negamax search (current_depth > 0)
        if current_depth > 0 and current_depth < len(self.killer_moves):
            if self.killer_moves[current_depth][0] == move:
                return 80000 
            if self.killer_moves[current_depth][1] == move:
                return 70000 
        
        # --- MVV/LVA for Captures ---
        if board.is_capture(move):
            attacker_piece = board.piece_at(move.from_square)
            if attacker_piece is None: # Should not happen with legal moves
                return score
            attacker = attacker_piece.piece_type 
            
            if board.is_en_passant(move):
                victim = chess.PAWN
            else:
                victim_piece = board.piece_at(move.to_square)
                if victim_piece is None: # Should not happen with captures
                    return score
                victim = victim_piece.piece_type 
                
            # MVV_LVA only supports P, N, B, R, Q (1-5), not King (6). Max attacker is Queen (5)
            if victim < 6 and attacker < 6:
                # Add a high value (100,000) to distinguish it clearly from non-capture scores
                return 100000 + self.MVV_LVA[victim-1, attacker-1]
            else:
                # Fallback for King/Queen captures that might violate the array indexing
                return 100000 + 55 
        
        # --- Other Heuristics ---
        if board.gives_check(move):
            score += 500
            
        piece = board.piece_at(move.from_square)
        if piece is not None and piece.piece_type == chess.KING:
            if board.is_castling(move):
                score += 200
            else:
                # Small penalty for unnecessary King moves early on
                if board.fullmove_number < 20:
                    score -= 50
                        
        return score
    
    def quiescence_search(self, board, alpha, beta, color):
        
        # NOTE: For QS TT entries, we treat depth as 0 (for the stand-pat evaluation)
        board_key = board.fen()
        original_alpha = alpha

        # 1. --- TT Lookup in Quiescence Search ---
        tt_entry = self.transposition_table.get(board_key)
        if tt_entry and tt_entry['depth'] >= 0: # Check if entry is useful (depth 0 means stand-pat score)
             if tt_entry['flag'] == 'EXACT':
                return tt_entry['score']

        # --- Terminal Node Check ---
        if board.is_checkmate():
            return self.MATE_SCORE - 1 
        elif board.is_stalemate():
            return 0
        
        # 2. Stand-pat Evaluation
        stand_pat = color * self.get_evaluation(board)
        
        max_score = stand_pat
        
        if max_score >= beta:
            # --- TT Store (Fail-High - Beta Cutoff) ---
            self.transposition_table[board_key] = {
                'score': max_score, 
                'depth': 0, 
                'flag': 'ALPHA', 
                'best_move': None 
            }
            return beta
            
        # Update alpha with the best non-tactical score so far
        if max_score > alpha:
            alpha = max_score

        # 3. Generate and Score Only Noisy Moves (Captures and Checks)
        # OPTIMIZATION: Use score_move (MVV/LVA) to prioritize the most promising tactical lines.
        scored_noisy_moves = []
        for move in board.legal_moves:
            if board.is_capture(move) or board.gives_check(move):
                # Use score_move logic (primarily MVV/LVA)
                move_score = self.score_move(board, move, 0) 
                scored_noisy_moves.append((move_score, move))
        
        # Sort by score in descending order (highest score first)
        scored_noisy_moves.sort(key=lambda x: x[0], reverse=True)
        
        # 4. Iterate through sorted noisy moves
        for move_score, move in scored_noisy_moves:
            
            # CRITICAL EFFICIENCY FIX: Use push/pop, DO NOT use board.copy()
            board.push(move)
            
            # Recursive call to Quiescence Search
            score = -self.quiescence_search(board, -beta, -alpha, -color)
            
            board.pop()
            
            max_score = max(max_score, score)
            
            # Alpha-Beta Cutoff within Quiescence Search
            if max_score >= beta:
                # --- TT Store (Fail-High - Beta Cutoff) ---
                self.transposition_table[board_key] = {
                    'score': max_score, 
                    'depth': 0, 
                    'flag': 'ALPHA', 
                    'best_move': move 
                }
                return beta
            alpha = max(alpha, max_score)
            
        # 5. --- TT Store (Exact/Fail-Low) ---
        tt_flag = 'EXACT' if max_score > original_alpha else 'BETA'
        self.transposition_table[board_key] = {
            'score': max_score, 
            'depth': 0, 
            'flag': tt_flag,
            'best_move': None # No best move needed for terminal QS
        }
        
        return max_score
      
        
    def negamax(self, board, depth, alpha, beta, color):
        
        # 1. --- Check for Terminal Nodes / Base Case FIRST ---
        
        # OPTIMIZATION: Prune branches that lead to known draws (repetition or 50-move rule)
        if board.is_fifty_moves() or board.is_repetition():
            return 0
        
        if board.is_checkmate():
            return self.MATE_SCORE - depth
        elif board.is_stalemate():
            return 0
        
        if depth == 0:
            return self.quiescence_search(board, alpha, beta, color)
        
        original_alpha = alpha
        board_key = board.fen()
        
        # 2. --- TT-lookup ---
        tt_entry = self.transposition_table.get(board_key)
        if tt_entry and tt_entry['depth'] >= depth:
            # Check if the stored score is sufficient for a cutoff or exact
            if tt_entry['flag'] == 'EXACT':
                return tt_entry['score']
            if tt_entry['flag'] == 'ALPHA' and tt_entry['score'] >= beta:
                return tt_entry['score'] # Fail-high cutoff
            if tt_entry['flag'] == 'BETA' and tt_entry['score'] <= alpha:
                return tt_entry['score'] # Fail-low cutoff
            
        # --- Adjust bounds for Mate Distance ---
        if alpha >= self.MATE_SCORE - 1:
            return alpha
        alpha = max(alpha, -self.MATE_SCORE + depth)
        
        max_score = -np.inf
        best_move_found = None
        
        moves = list(board.legal_moves)
        
        # If no legal moves remain, it must be mate/stalemate (should be caught above, but as a safeguard)
        if not moves:
            return 0 

        # --- Move Ordering ---
        scored_moves = []
        for move in moves:
            move_score = self.score_move(board, move, depth)
            scored_moves.append((move_score, move))
            
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        
        for move_score,move in scored_moves:
            
            # CRITICAL EFFICIENCY FIX: Use push/pop, DO NOT use board.copy()
            board.push(move)
            
            # Pass the same board instance, as the state is managed by push/pop
            score = -self.negamax(board, depth - 1, -beta, -alpha, -color)
            
            board.pop()
            
            if score > max_score:
                max_score = score
                best_move_found = move
            alpha = max(alpha, max_score)
            
            if alpha >= beta:
                # Alpha-beta cutoff
                # --- Killer Move Storage ---
                if depth < len(self.killer_moves):
                    self.killer_moves[depth][1] = self.killer_moves[depth][0]
                    self.killer_moves[depth][0] = move
                
                # --- TT Storage (Fail-High) ---
                tt_flag = 'ALPHA' 
                self.transposition_table[board_key] = {
                    'score': max_score, 
                    'depth': depth, 
                    'flag': tt_flag,
                    'best_move': move 
                }
                break
            
        # --- TT Storage (Exact/Fail-Low) ---
        tt_flag = 'EXACT' if max_score > original_alpha else 'BETA'
        
        if best_move_found is not None:
             self.transposition_table[board_key] = {
                'score': max_score, 
                'depth': depth, 
                'flag': tt_flag,
                'best_move': best_move_found
            }
                
        return max_score
    
    def get_evaluation(self, board):
        """
        Converts the chess.Board state into the required tensor format,
        scales the features, feeds it to the PyTorch CNN, and returns 
        the rescaled, final evaluation score.
        """
        material_imbalance = b.get_material_imbalance(board)
        board_features = b.get_board_channels(board)
    
        # Prepare board features for PyTorch: (Batch, Channels, Height, Width)
        X = board_features.reshape(1, self.n_input_channels, 8, 8) 
        
        X_scaled = np.empty_like(X) # Not strictly used, but keeps the original code structure
        X_extra = np.array([material_imbalance]).reshape(1,-1) 
        
        # Only normalize the specified channels (18 and 19 based on your original code)
        norm_channels = [18,19]
    
        for idx,channel in enumerate(norm_channels):
            X_channel_flat = X[:, channel, :, :].reshape(X.shape[0], -1)           
            X_scaled_flat = self.scalers_board[idx].transform(X_channel_flat)
            # Apply the scaled data back to the X array
            X[:, channel, :, :] = X_scaled_flat.reshape(X[:, channel, :, :].shape)
           
        # Convert numpy to PyTorch tensor
        X = torch.from_numpy(X).float()
        
        X_extra_scaled = self.scaler_extra.transform(X_extra)
        X_extra = torch.from_numpy(X_extra_scaled).float()
           
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Inference
        with torch.no_grad():
            evaluation = self.model(X,X_extra).detach().numpy().reshape(1,-1)
            
        # Rescale the output score
        evaluation_rescaled = self.scaler_y.inverse_transform(evaluation)
    
        # Return score, divided by 100 to convert from centipawns (if that was the training scale)
        return evaluation_rescaled[0,0]/100 
    
    def get_lichess_opening_data(self, fen):
        """Fetches Lichess Master opening data for the current FEN."""
        try:
            url = "https://explorer.lichess.ovh/masters"
            params = {
                "fen": fen,
                'moves': 5,
                "topGames": 0  # We only need the opening moves, not full games
            }
            
            response = requests.get(url, params=params, timeout=2) # Add timeout
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
            
            return response.json()
        
        except requests.exceptions.RequestException as e:
            # print(f"Error fetching data from Lichess API: {e}") # Suppress this print in engine loop
            return None             
    

if __name__ == '__main__':
    
    # --- Initialization ---
    engine_path = 'chess_cnn_model.pth'
           
    scalers_board_filename = 'scalers_board.joblib'
    scaler_extra_filename = 'scaler_extra.joblib'
    scaler_y_filename = 'scaler_y.joblib'

    # Note: Requires 'scalers/' directory and the specified files to exist, 
    # as well as the 'board_utils.py' module and 'chess_cnn_model.pth'.
    try:
        engine = Engine(engine_path,scalers_board_filename,scaler_extra_filename,scaler_y_filename)
    except Exception as e:
        print(f"ERROR: Failed to initialize Engine. Check if model and scaler files exist in the correct paths.")
        print(f"Details: {e}")
        # Exit gracefully if initialization fails
        exit()
    
    board = chess.Board()  
    
    # Set the user to play as black (White=True, Black=False).
    # If board.turn is True (White), user_col should be False (Black).
    # Since board.turn starts at True, user_col should be False (or chess.BLACK) for the engine to move first.
    user_col = 1
    
    print("Welcome to the Chess Engine!")
    print("You are playing as Black.")
    print(board)

    # --- Game Loop ---
    while not board.is_game_over():
        
        if board.turn == user_col:
            # User's turn
            move = user_go(board)
            
        else:
            # Engine's turn (White)
            print("Engine is thinking...")
            # CRITICAL: Pass a copy to FindBestMove so the engine doesn't modify the main game board
            move = engine.find_best_move(board.copy()) 
            print(f"Engine plays: {move.uci()}")
            
        if move:
            board.push(move)
            print("\n" + "="*20)
            print(board)
            print("="*20 + "\n")
        else:
            print("ERROR: Engine failed to find a move or game is over.")
            break

    print("GAME OVER")
    print(f"Result: {board.result()}")
