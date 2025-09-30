import chess
import lichess

import joblib
import torch

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
        
        self.scaler_board = joblib.load("scalers/"+scalers_board_filename)
        self.scaler_extra = joblib.load("scalers/"+scaler_extra_filename)
        self.scaler_y = joblib.load("scalers/"+scaler_y_filename)
        
        self.n_input_channels = 20
        self.extra_features_size = 1
        self.model = b.load_model(b.CNN_Regressor, engine_path, self.n_input_channels, self.extra_features_size)
        
        self.transposition_table = {}
        self.killer_moves = [[None, None] for _ in range(100)]
        

    def minimax(self,depth,move,maximizingPlayer,alpha,beta,board,model,n_input_channels,scalers_board,scaler_extra,scaler_y): 
        
        moves = {}
        
        if depth == 4:
            evaluation = self.get_evaluation(board.copy())
            return evaluation
        
        if maximizingPlayer: 
         
            best = -np.inf
     
            for move in board.legal_moves: 
                board.push(move)
                val = self.minimax(depth + 1, move, False, alpha, beta,board.copy()) 
                board.pop()
                    
                best = max(best, val) 
                alpha = max(alpha, best) 
                
                if depth == 0:
                    moves[move] = val
    
                # Alpha Beta Pruning 
                if beta <= alpha: 
                    break 
            if depth == 0:
                return moves
            return best
         
        else:
            best = np.inf
    
            for move in board.legal_moves: 
                board.push(move)
                val = self.minimax(depth + 1, move, True, alpha, beta,board.copy()) 
                board.pop()
            
                best = min(best, val) 
                beta = min(beta, best) 
                
                if depth == 0:
                    moves[move] = val
    
                # Alpha Beta Pruning 
                if beta <= alpha: 
                    break 
            if depth == 0:
                return moves
            return best
    
    def get_evaluation(self,board):
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
    
    
    def move(self,board,model,n_input_channels,scalers_board,scaler_extra,scaler_y):
        
        if board.fullmove_number < 5:
            fen = board.fen()
            opening_data = self.get_lichess_opening_data(fen)
            n_moves = len(opening_data['moves'])
            if n_moves > 0:
                move_choice = np.random.randint(0,n_moves)
                move = opening_data['moves'][move_choice]['uci']
                return chess.Move.from_uci(move)
        
        moves = self.minimax(0,None,board.turn,-np.inf,np.inf,board)
        
        if board.turn:
            move = max(moves,key=moves.get)
        else:
            move = min(moves,key=moves.get)
        print(moves)
        print("Best Move:",move)
        return move
    
    def get_lichess_opening_data(self,fen):
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
       
engine_path = 'chess_cnn_model.pth'
       
scalers_board_filename = 'scalers_board.joblib'
scaler_extra_filename = 'scaler_extra.joblib'
scaler_y_filename = 'scaler_y.joblib'

engine = Engine(engine_path,scalers_board_filename,scaler_extra_filename,scaler_y_filename)

    

if __name__ == '__main__':
    
    board = chess.Board()  
    user_col = 1
    
    while not board.is_checkmate():
        if board.turn == user_col:
            move = user_go(board)
        else:
            move = engine.move(board.copy())
            
        board.push(move)
        print(board)
    print("GAME END")