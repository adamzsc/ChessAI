import chess
import lichess

import joblib
import torch
import time

import numpy as np

import board_utils as b
import requests

def user_go(board):
    while True:
        move_input = input("Enter Move: ")
        move = chess.Move.from_uci(move_input)
        if move in board.legal_moves:
            return move
        else:
            print("NOT A LEGAL MOVE")

class Engine():
    
    def __init__(self,engine_path,scalers_board_filename,scaler_extra_filename,scaler_y_filename):
        
        self.scalers_board = joblib.load("scalers/"+scalers_board_filename)
        self.scaler_extra = joblib.load("scalers/"+scaler_extra_filename)
        self.scaler_y = joblib.load("scalers/"+scaler_y_filename)
        
        self.n_input_channels = 20
        self.extra_features_size = 1
        self.model = b.load_model(b.CNN_Regressor, engine_path, self.n_input_channels, self.extra_features_size)
        
        self.transposition_table = {}
        self.killer_moves = [[None, None] for _ in range(100)]    
        self.principal_variation_move = None 
        
        self.MATE_SCORE = 1000
        
        self.MVV_LVA =  np.array([   #accesed  [victim_type - 1][attacker_type - 1]
            [15, 14, 13, 12, 11, 10], # victim P, attacker P, N, B, R, Q, K
            [25, 24, 23, 22, 21, 20], # victim N, attacker P, N, B, R, Q, K   
            [35, 34, 33, 32, 31, 30], # victim B, attacker P, N, B, R, Q, K
            [45 ,44, 43, 42, 41, 40], # victim R, attacker P, N, B, R, Q, K
            [55, 54, 53, 52, 51, 50], # victim Q, attacker P, N, B, R, Q, K
            ] )    
        
    def find_best_move(self, board, time_limit_seconds=10.0):             
        
        #access database for  opening 
        if board.fullmove_number < 5:
            fen = board.fen()
            opening_data = self.get_lichess_opening_data(fen)
            n_moves = len(opening_data['moves'])
            if n_moves > 0:
                move_choice = np.random.randint(0,n_moves)
                move = opening_data['moves'][move_choice]['uci']
                return chess.Move.from_uci(move)
            
        start_time = time.time()
        current_best_move = None
        
        for depth in range(1,100):
            
            if (time.time() - start_time) >= time_limit_seconds:
                break
        
            print(f"\n--- Starting Depth {depth} ---")
        
            best_score_this_depth = -np.inf
            best_move_this_depth = None
            alpha = -np.inf
            beta = np.inf
            
            if board.turn == 1:
                color = 1
            else:
                color = -1

            moves = list(board.legal_moves)        
            scored_moves = []
            
            for move in moves:
                move_score = self.score_move(board, move, depth)
                scored_moves.append((move_score, move))
                
            scored_moves.sort(key=lambda x: x[0], reverse=True)
            
            #Iterate through sorted moves
            for move_score, move in scored_moves:                
                
                if (time.time() - start_time) >= time_limit_seconds:
                    print(f"Stopping search at depth {depth} due to time limit.")
                    return current_best_move
                
                board.push(move)
            
                score = -self.negamax(board,depth - 1,-beta,-alpha,-color)
                    
                board.pop()
                
                if score > best_score_this_depth:
                    best_score_this_depth = score
                    best_move_this_depth = move
                    
                alpha = max(alpha, best_score_this_depth) # Update the root alpha value
                current_best_move = best_move_this_depth
                # Store the PV move for Move Ordering in the next iteration
                self.principal_variation_move = current_best_move 
            
                print(f"Depth {depth} completed. Best Move: {current_best_move} | Score: {best_score_this_depth:.2f}")

        # Return the best move found in the last full iteration
        return current_best_move
        
    def score_move(self,board,move,current_depth):
        
        score = 100
        
        if move == self.principal_variation_move:
            return 100000
        
        if current_depth < len(self.killer_moves):
            if self.killer_moves[current_depth][0] == move:
                return 80000 # Primary Killer
            if self.killer_moves[current_depth][1] == move:
                return 70000 # Secondary Killer
        
        #check if capture
        if board.is_capture(move):
            attacker = board.piece_at(move.from_square).piece_type 
            if board.is_en_passant(move):
                victim = chess.PAWN
            else:
                victim = board.piece_at(move.to_square).piece_type 
                
            score += self.MVV_LVA[victim-1,attacker-1]
        
        if board.gives_check(move):
            score += 500
            
        if board.piece_at(move.from_square).piece_type == 6:
            if board.is_castling(move):
                score += 200
            else:
                if board.fullmove_number < 20:
                    score -= 200
                        
        return score
    
    def quiescence_search(self, board, alpha, beta, color):
        
        if board.is_checkmate():
            return self.MATE_SCORE - 1
        elif board.is_stalemate():
            return 0
        
        board_key = board.fen()
        original_alpha = alpha

        # TT Lookup in Quiescence Search
        tt_entry = self.transposition_table.get(board_key)
        if tt_entry and tt_entry['depth'] >= 0: # Check if entry is useful (depth 0 means stand-pat score)
             if tt_entry['flag'] == 'EXACT':
                return tt_entry['score']
             # TT entries from the main negamax search (depth > 0) are not directly reusable here
             # as they might cut off a tactical sequence that QS needs to explore.
        
        stand_pat = color * self.get_evaluation(board)
        
        max_score = stand_pat
        
        if max_score >= beta:
            return beta
            
        # Update alpha with the best non-tactical score so far
        if max_score > alpha:
            alpha = max_score

        #Iterate through noisy moves
        for move in board.legal_moves:
            
            if board.is_capture(move) or board.gives_check(move):
            
                board.push(move)
                
                # Recursive call to Quiescence Search (NOT negamax)
                score = -self.quiescence_search(board, -beta, -alpha, -color)
                
                max_score = max(max_score, score)
                
                board.pop()
                
                # Alpha-Beta Cutoff within Quiescence Search
                if max_score >= beta:
                    return beta
                alpha = max(alpha, max_score)
                
        #TT Store (Exact/Fail-Low) ---
        tt_flag = 'EXACT' if max_score > original_alpha else 'BETA'
        self.transposition_table[board_key] = {
            'score': max_score, 
            'depth': 0, 
            'flag': tt_flag,
            'best_move': None # No best move needed for terminal QS
        }
            
        return max_score
      
        
    def negamax(self, board, depth, alpha, beta, color):
        
        #Check for checkmate/stalemate 
        
        if board.is_checkmate():
            return self.MATE_SCORE - depth
        elif board.is_stalemate():
            return 0
                
        original_alpha = alpha
        board_key = board.fen()
        
        #TT-lookup
        
        tt_entry = self.transposition_table.get(board_key)
        if tt_entry and tt_entry['depth'] >= depth:
            # Check if the stored score is sufficient for a cutoff or exact
            if tt_entry['flag'] == 'EXACT':
                return tt_entry['score']
            if tt_entry['flag'] == 'ALPHA' and tt_entry['score'] >= beta:
                return tt_entry['score'] # Fail-high cutoff
            if tt_entry['flag'] == 'BETA' and tt_entry['score'] <= alpha:
                return tt_entry['score'] # Fail-low cutoff       
        
        if depth == 0:
            return self.quiescence_search(board, alpha, beta, color)
            #return color * self.get_evaluation(board)
        
        if alpha >= self.MATE_SCORE - 1:
            return alpha
        alpha = max(alpha, -self.MATE_SCORE + depth)
        
        max_score = -np.inf
        best_move_found = None
        
        moves = list(board.legal_moves)
        scored_moves = []
        for move in moves:
            move_score = self.score_move(board, move, depth)
            scored_moves.append((move_score, move))
            
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        
        for move_score,move in scored_moves:
            board.push(move)
            
            score = -self.negamax(board, depth - 1, -beta, -alpha, -color)
            
            board.pop()
            
            if score > max_score:
                max_score = score
                best_move_found = move
            alpha = max(alpha, max_score)
            
            if alpha >= beta:
                # Alpha-beta cutoff
                if depth < len(self.killer_moves):
                    # Store this move as the primary killer for this depth
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
            
        #Transposition Table Storage (Exact/Fail-Low)
        
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
           
        evaluation = self.model(X,X_extra).detach().numpy().reshape(1,-1)
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
    

if __name__ == '__main__':
    
    engine_path = 'chess_cnn_model.pth'
           
    scalers_board_filename = 'scalers_board.joblib'
    scaler_extra_filename = 'scaler_extra.joblib'
    scaler_y_filename = 'scaler_y.joblib'

    engine = Engine(engine_path,scalers_board_filename,scaler_extra_filename,scaler_y_filename)
    
    board = chess.Board()  
    user_col = 1
    
    while not board.is_checkmate():
        if board.turn == user_col:
            move = user_go(board)
        else:
            move = engine.find_best_move(board.copy())
            
        board.push(move)
        print(board)
    print("GAME END")