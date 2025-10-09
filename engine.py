import chess
import chess.polyglot

import joblib
import board_utils as b
import requests

import numpy as np
import time
import torch

class Engine():
    
    def __init__(self,model_path,use_opening_table = True,max_depth=20,time_limit=10):
    
        scalers_board_filename = 'scalers_board.joblib'
        scaler_extra_filename = 'scaler_extra.joblib'
        scaler_y_filename = 'scaler_y.joblib'
        
        self.scalers_board = joblib.load("scalers/"+scalers_board_filename)
        self.scaler_extra = joblib.load("scalers/"+scaler_extra_filename)
        self.scaler_y = joblib.load("scalers/"+scaler_y_filename)
        
        self.n_input_channels = 20
        self.extra_features_size = 1
        self.model = b.load_model(b.CNN_Regressor, model_path, self.n_input_channels, self.extra_features_size)
        
        self.transposition_table = {}
        # Killer moves are stored by depth index: [[Killer1, Killer2], [Killer1, Killer2], ...]
        self.killer_moves = [[None, None] for _ in range(100)]    
        self.principal_variation_move = None
        # History table stores cutoff scores for quiet moves: [from_square][to_square]
        self.history_table = np.zeros((64, 64), dtype=np.int32)
        
        self.MATE_SCORE = 10000
        
        self.nodes_searched = 0
        
        self.max_depth = max_depth
        self.time_limit = time_limit
        
        self.repetition_counter = {}
        
        # MVV/LVA scoring table
        self.MVV_LVA =  np.array([   #accesed  [victim_type - 1][attacker_type - 1]
            [15, 14, 13, 12, 11, 10], # victim P, attacker P, N, B, R, Q, K
            [25, 24, 23, 22, 21, 20], # victim N, attacker P, N, B, R, Q, K   
            [35, 34, 33, 32, 31, 30], # victim B, attacker P, N, B, R, Q, K
            [45 ,44, 43, 42, 41, 40], # victim R, attacker P, N, B, R, Q, K
            [55, 54, 53, 52, 51, 50], # victim Q, attacker P, N, B, R, Q, K
            ])    
        
    def find_best_move(self, board):       
        
        self.nodes_searched = 0
        
        board_key = chess.polyglot.zobrist_hash(board)       
        self.repetition_counter[board_key] = self.repetition_counter.get(board_key, 0) + 1
        
        if self.use_opening_table:
            # Access database for opening
            if board.fullmove_number < 5:
                fen = board.fen()
                opening_data = self.get_lichess_opening_data(fen)
                n_moves = len(opening_data['moves'])
                if n_moves > 0:
                    if n_moves < 3:
                        move_choice = np.random.randint(0,n_moves)                   
                    elif n_moves >= 3:
                        move_choice = np.random.randint(0,3)
                    move = opening_data['moves'][move_choice]['uci']
                    return chess.Move.from_uci(move)
            
        start_time = time.time()
        current_best_move = None
        self.history_table.fill(0) # Clear history table at the start of the full search
        
        for depth in range(1,self.max_depth):
            
            if (time.time() - start_time) >= self.time_limit:
                break
        
            #print(f"\n--- Starting Depth {depth} ---")
        
            best_score_this_depth = -np.inf
            best_move_this_depth = None
            alpha = -np.inf
            beta = np.inf
            
            color = 1 if board.turn == chess.WHITE else -1

            moves = list(board.legal_moves)        
            scored_moves = []
            
            # Note: No TT move available at the root search before the first negamax call
            for move in moves:
                # Only PV move, Killers, History, and MVV/LVA apply here
                move_score = self.score_move(board, move, depth, tt_move=None) 
                scored_moves.append((move_score, move))
                
            scored_moves.sort(key=lambda x: x[0], reverse=True)
            
            # Iterate through sorted moves
            for move_score, move in scored_moves:                
                
                if (time.time() - start_time) >= self.time_limit:
                    break
                
                board.push(move)
            
                # Standard full-window search at the root
                score = -self.negamax(board, depth - 1, -beta, -alpha, -color)
                    
                board.pop()
                
                if score > best_score_this_depth:
                    best_score_this_depth = score
                    best_move_this_depth = move
                    
                alpha = max(alpha, best_score_this_depth) # Update the root alpha value
                current_best_move = best_move_this_depth
                # Store the PV move for Move Ordering in the next iteration
                self.principal_variation_move = current_best_move                 
            
        # Return the best move found in the last full iteration
        
        #print("Nodes Searched:", self.nodes_searched)
        nps = self.nodes_searched/self.time_limit
        #print("Avg. ",nps,"Nodes/s")
        
        board.push(current_best_move)
        
        new_hash = chess.polyglot.zobrist_hash(board)       
        self.repetition_counter[new_hash] = self.repetition_counter.get(new_hash, 0) + 1        
        return current_best_move
        
    def score_move(self,board,move,current_depth,tt_move = None):
        
        score = 500
        
        # 1. Transposition Table Move (Highest Priority)
        if move == tt_move:
             return 10000000 
        
        # 2. Principal Variation Move (Second Highest Priority - only applies if explicitly set)
        if move == self.principal_variation_move:
            return 9000000 
        
        # 3. Killer Moves (For non-captures that caused a cutoff in the *same* depth)
        if current_depth > 0 and current_depth < len(self.killer_moves):
            if move == self.killer_moves[current_depth][0]:
                return 80000 # Primary killer
            if move == self.killer_moves[current_depth][1]:
                return 70000 # Secondary killer
        
        # 4. MVV/LVA (Most Valuable Victim, Least Valuable Attacker) - for captures
        if board.is_capture(move):
            attacker_piece = board.piece_at(move.from_square)
            if attacker_piece is None: 
                return score
            attacker = attacker_piece.piece_type 
            
            if board.is_en_passant(move):
                victim = chess.PAWN
            else:
                victim_piece = board.piece_at(move.to_square)
                if victim_piece is None: 
                    return score
                victim = victim_piece.piece_type 
                
            score += (500 + self.MVV_LVA[victim-1, attacker-1])
        
        # 5. History Heuristic (For quiet moves that have caused cutoffs in the past)
        if not board.is_capture(move) and not board.gives_check(move):
            # The History table score (a large integer) is added directly
            score += self.history_table[move.from_square][move.to_square]
        
        # 6. Other Heuristics (Lower Priority)
        if board.gives_check(move):
            score += 500
            
        piece = board.piece_at(move.from_square)
        if piece is not None and piece.piece_type == chess.KING:
            if board.is_castling(move):
                score += 200
            else:
                if board.fullmove_number < 20:
                    score -= 50
                        
        return score
    
    def quiescence_search(self, board, alpha, beta, color, mate_ply_extension):
        
        self.nodes_searched += 1
        
        board_key = chess.polyglot.zobrist_hash(board)
        
        if board.is_checkmate():
            return -(self.MATE_SCORE + mate_ply_extension)
        elif board.is_stalemate() or self.repetition_counter.get(board_key, 0) >= 2:
            return 0
        
        self.repetition_counter[board_key] = self.repetition_counter.get(board_key, 0) + 1
        
        try:

            original_alpha = alpha
    
            # TT Lookup in Quiescence Search (QS depth is 0)
            tt_entry = self.transposition_table.get(board_key)
            if tt_entry and tt_entry['depth'] >= 0: 
                 if tt_entry['flag'] == 'EXACT':
                     return tt_entry['score']
    
            # Stand-Pat (The score if we make no tactical moves)
            stand_pat = color * self.get_evaluation(board)
            
            max_score = stand_pat
            
            if max_score >= beta:
                return beta
                
            # Update alpha with the best non-tactical score so far
            if max_score > alpha:
                alpha = max_score
    
            scored_noisy_moves = []
            # Generate moves that are either captures or checks
            for move in board.legal_moves:
                if board.is_capture(move) or board.gives_check(move):
                    # Use score_move logic (primarily MVV/LVA)
                    move_score = self.score_move(board, move, 0, tt_move=None) 
                    scored_noisy_moves.append((move_score, move))
            
            # Sort by score in descending order (highest score first)
            scored_noisy_moves.sort(key=lambda x: x[0], reverse=True)
            
            # Iterate through sorted noisy moves
            for move_score, move in scored_noisy_moves:
                
                board.push(move)
                
                # Recursive call to Quiescence Search (NOT negamax)
                score = -self.quiescence_search(board, -beta, -alpha, -color, mate_ply_extension + 1)
                
                max_score = max(max_score, score)
                
                board.pop()
                
                # Alpha-Beta Cutoff within Quiescence Search
                if max_score >= beta:
                    return beta
                alpha = max(alpha, max_score)
                    
            # TT Store (Exact/Fail-Low)
            tt_flag = 'EXACT' if max_score > original_alpha else 'BETA'
            self.transposition_table[board_key] = {
                'score': max_score, 
                'depth': 0, 
                'flag': tt_flag,
                'best_move': None # No best move needed for terminal QS
            }
                
            return max_score
        
        finally:
            self.repetition_counter[board_key] -= 1
            if self.repetition_counter[board_key] == 0:
                del self.repetition_counter[board_key]    
      
        
    def negamax(self, board, depth, alpha, beta, color):
        
        self.nodes_searched += 1
        
        board_key = chess.polyglot.zobrist_hash(board)
        
        if board.is_checkmate():
            return -(self.MATE_SCORE + depth)
        elif board.is_stalemate() or self.repetition_counter.get(board_key, 0) >= 2:
            return 0
        
        self.repetition_counter[board_key] = self.repetition_counter.get(board_key, 0) + 1
        
        try:
                
            original_alpha = alpha
            
            # TT-lookup
            tt_move = None
            tt_entry = self.transposition_table.get(board_key)
            if tt_entry and tt_entry['depth'] >= depth:
                # Check if the stored score is sufficient for a cutoff or exact
                if tt_entry['flag'] == 'EXACT':
                    return tt_entry['score']
                if tt_entry['flag'] == 'ALPHA' and tt_entry['score'] >= beta:
                    return tt_entry['score'] # Fail-high cutoff
                if tt_entry['flag'] == 'BETA' and tt_entry['score'] <= alpha:
                    return tt_entry['score'] # Fail-low cutoff    
                
                if tt_entry.get('best_move'):
                     tt_move = tt_entry['best_move']
            
            # Base case: Search ends, call Quiescence Search
            if depth == 0:
                return self.quiescence_search(board, alpha, beta, color, 0)
            
            # Mate score adjustment
            if alpha >= self.MATE_SCORE - 1:
                return alpha
            alpha = max(alpha, -self.MATE_SCORE + depth)
            
            max_score = -np.inf
            best_move_found = None
            
            # Move Ordering: Score and Sort
            moves = list(board.legal_moves)
            scored_moves = []
            for move in moves:
                # FIX: Pass the TT move here for high priority!
                move_score = self.score_move(board, move, depth, tt_move)
                scored_moves.append((move_score, move))
                
            scored_moves.sort(key=lambda x: x[0], reverse=True)
            
            move_index = 0 # Track move order for LMR
            
            for move_score,move in scored_moves:
                
                search_depth = depth - 1
                reduction = 0
                
                # --- LATE MOVE REDUCTION (LMR) LOGIC ---
                is_quiet = not board.is_capture(move)
                is_check = board.gives_check(move)
    
                # Apply LMR to quiet moves that are searched late, at a reasonable depth
                if is_quiet and not is_check and move_index >= 4 and depth >= 3:
                    reduction = 1 
                    
                    # Deeper reduction for moves even later in the list and at higher depths
                    if move_index >= 8 and depth >= 5:
                        reduction = 2
                            
                    # Ensure we don't reduce below depth 1
                    reduction = min(reduction, depth - 1)
                    
                    search_depth = depth - 1 - reduction
                
                board.push(move)
                
                # --- Perform Search ---
                if reduction > 0:
                    # 1. Reduced Search (Null Window PVS for efficiency)
                    # Search [alpha, alpha + 1]. If score > alpha, it is a PV candidate.
                    score = -self.negamax(board, search_depth, -(alpha + 1), -alpha, -color)
                    
                    # 2. Re-search at full depth if the reduced search was promising (failed high)
                    if score > alpha: 
                        score = -self.negamax(board, depth - 1, -beta, -alpha, -color)
                else:
                    # 3. Full-depth search for prioritized moves
                    score = -self.negamax(board, depth - 1, -beta, -alpha, -color)
                
                board.pop() # Undo the move
                
                if score > max_score:
                    max_score = score
                    best_move_found = move
                alpha = max(alpha, max_score)
                
                if alpha >= beta:
                    # Alpha-beta cutoff (Fail-High)
                    
                    # --- History Update ---
                    if not board.is_capture(move):
                        # Add a score proportional to the depth^2 for a strong history effect
                        self.history_table[move.from_square][move.to_square] += depth ** 2
    
                    # --- Killer Move Storage ---
                    if not board.is_capture(move) and depth < len(self.killer_moves):
                        # Shift the second best killer to the third spot (or discard) and insert new best
                        self.killer_moves[depth][1] = self.killer_moves[depth][0]
                        self.killer_moves[depth][0] = move
                    
                    # Transposition Table Storage (Fail-High)
                    tt_flag = 'ALPHA' 
                    self.transposition_table[board_key] = {
                        'score': max_score, 
                        'depth': depth, 
                        'flag': tt_flag,
                        'best_move': move 
                    }
                    break
                
                move_index += 1 # Increment index for LMR tracking
                
            # Transposition Table Storage (Exact/Fail-Low)
            tt_flag = 'EXACT' if max_score > original_alpha else 'BETA'
            
            if best_move_found is not None:
                 self.transposition_table[board_key] = {
                    'score': max_score, 
                    'depth': depth, 
                    'flag': tt_flag,
                    'best_move': best_move_found
                }
                    
            return max_score
        
        finally:
            self.repetition_counter[board_key] -= 1
            if self.repetition_counter[board_key] == 0:
                del self.repetition_counter[board_key]
    
    def get_evaluation(self, board):

        material_imbalance = b.get_material_imbalance(board)
        board_features = b.get_board_channels(board)
    
        X = board_features.reshape(1, self.n_input_channels, 8, 8) 
        
        X_scaled = np.empty_like(X)
        X_extra = np.array([material_imbalance]).reshape(1,-1) 
        
        norm_channels = [18,19]
    
        for idx,channel in enumerate(norm_channels):
            X_channel_flat = X[:, channel, :, :].reshape(X.shape[0], -1)           
            X_scaled_flat = self.scalers_board[idx].transform(X_channel_flat)
            X[:, channel, :, :] = X_scaled_flat.reshape(X_scaled[:, channel, :, :].shape)
           
        X = torch.from_numpy(X).float()
        
        X_extra_scaled = self.scaler_extra.transform(X_extra)
        X_extra = torch.from_numpy(X_extra_scaled).float()
          
        evaluation = [[self.model(X,X_extra).item()]]
        evaluation_rescaled = self.scaler_y.inverse_transform(evaluation)
    
        return evaluation_rescaled[0,0]/100
    
    def get_lichess_opening_data(self, fen):
        try:
            url = "https://explorer.lichess.ovh/masters"
            params = {
                "fen": fen,
                'moves': 5,
                "topGames": 0  # We only need the opening moves, not full games
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
            
            return response.json()
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from Lichess API: {e}")
            return None             
    
