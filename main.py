import chess
import chess.svg

import joblib
import torch

import numpy as np
import matplotlib.pyplot as plt

import board_utils as b

def user_go():
    move_input = input("Enter Move: ")
    return chess.Move.from_uci(move_input)

def minimax(depth,move,maximizingPlayer,alpha,beta,board,model,n_input_channels,scalers_board,scaler_extra,scaler_y): 
    
    moves = {}
    
    if depth == 4:
        evaluation = get_evaluation(board.copy(),model,n_input_channels,scalers_board,scaler_extra,scaler_y)
        return evaluation
    
    if maximizingPlayer: 
     
        best = -np.inf

        for move in board.legal_moves: 
            board.push(move)
            val = minimax(depth + 1, move,
                          False, alpha, beta,board.copy(),model,n_input_channels,scalers_board,scaler_extra,scaler_y) 
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
            val = minimax(depth + 1, move,
                            True, alpha, beta,board.copy(),model,n_input_channels,scalers_board,scaler_extra,scaler_y) 
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

def get_evaluation(board,model,n_input_channels,scalers_board,scaler_extra,scaler_y):
    material_imbalance = b.get_material_imbalance(board)
    board_features = b.get_board_channels(board)

    X = board_features.reshape(1, n_input_channels, 8, 8) 
    
    X_scaled = np.empty_like(X)
    X_extra = np.array([material_imbalance]).reshape(1,-1) 
    
    norm_channels = [18,19]

    for idx,channel in enumerate(norm_channels):
        X_channel_flat = X[:, channel, :, :].reshape(X.shape[0], -1)           
        X_scaled_flat = scalers_board[idx].transform(X_channel_flat)
        X[:, channel, :, :] = X_scaled_flat.reshape(X_scaled[:, channel, :, :].shape)
       
    X = torch.from_numpy(X).float()
    
    X_extra_scaled = scaler_extra.transform(X_extra)
    X_extra = torch.from_numpy(X_extra_scaled).float()
       
    evaluation = model(X,X_extra).detach().numpy().reshape(1,-1)
    evaluation_rescaled = scaler_y.inverse_transform(evaluation)

    return evaluation_rescaled[0,0]/100

def engine_go(board,model,n_input_channels,scalers_board,scaler_extra,scaler_y):
    moves = minimax(0,None,board.turn,-np.inf,np.inf,board,model,n_input_channels,scalers_board,scaler_extra,scaler_y)
    move = min(moves,key=moves.get)
    print(moves)
    print("Best Move:",move)
    return move
    

if __name__ == '__main__':
    board = chess.Board()
    
    path = 'chess_cnn_model.pth'
    scalers_board_filename = 'scalers_board.joblib'
    scaler_extra_filename = 'scaler_extra.joblib'
    scaler_y_filename = 'scaler_y.joblib'
    
    scaler_board = joblib.load(scalers_board_filename)
    scaler_extra = joblib.load(scaler_extra_filename)
    scaler_y = joblib.load(scaler_y_filename)
    
    input_channels = 20
    extra_features_size = 1
    model = b.load_model(b.CNN_Regressor, path, input_channels, extra_features_size)
    
    while not board.is_checkmate():
        if board.turn:
            move = user_go()
        else:
            move = engine_go(board.copy(),model,input_channels,scaler_board,scaler_extra,scaler_y)
            
        board.push(move)
        print(board)
    print("GAME END")
