import torch
import torch.nn as nn

import chess
import numpy as np

class CNN_Regressor(nn.Module):
    def __init__(self, input_channels, extra_features_size):

        super(CNN_Regressor, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        
        cnn_output_size = 64 * 2 * 2
        fc1_input_size = cnn_output_size + extra_features_size
        
        # Dense layers for regression
        self.fc1 = nn.Linear(in_features=fc1_input_size, out_features=64)
        self.relu3 = nn.ReLU()
        self.output = nn.Linear(in_features=64, out_features=1) # Output for a single regression value

    def forward(self, x, extra_features):

        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        
        x = torch.cat((x, extra_features), dim=1)
        
        x = self.relu3(self.fc1(x))
        x = self.output(x)
        return x

def save_model(model, path):

    torch.save(model.state_dict(), "cnn_models/"+path)
    #print(f"Model saved successfully to {path}")

def load_model(model_class, path, input_channels, extra_features_size):

    model = model_class(input_channels, extra_features_size)
    model.load_state_dict(torch.load("cnn_models/"+path))
    model.eval()
    #print(f"Model loaded successfully from {path}")
    return model

def get_board_channels(board):
    row_array = np.array([],dtype='float32')
    
    row_array = np.concatenate((row_array,get_piece_channel(board,1,0)))
    row_array = np.concatenate((row_array,get_piece_channel(board,1,1)))
    
    row_array = np.concatenate((row_array,get_piece_channel(board,2,0)))
    row_array = np.concatenate((row_array,get_piece_channel(board,2,1)))
    
    row_array = np.concatenate((row_array,get_piece_channel(board,3,0)))
    row_array = np.concatenate((row_array,get_piece_channel(board,3,1)))
    
    row_array = np.concatenate((row_array,get_piece_channel(board,4,0)))
    row_array = np.concatenate((row_array,get_piece_channel(board,4,1)))
    
    row_array = np.concatenate((row_array,get_piece_channel(board,5,0)))
    row_array = np.concatenate((row_array,get_piece_channel(board,5,1)))
    
    row_array = np.concatenate((row_array,get_piece_channel(board,6,0)))
    row_array = np.concatenate((row_array,get_piece_channel(board,6,1)))
    
    row_array = np.concatenate((row_array,get_turn_to_move_channel(board)))
    
    row_array = np.concatenate((row_array,get_castling_channel(board,'kingside',1)))
    row_array = np.concatenate((row_array,get_castling_channel(board,'kingside',0)))
    row_array = np.concatenate((row_array,get_castling_channel(board,'queenside',0)))
    row_array = np.concatenate((row_array,get_castling_channel(board,'queenside',1)))
    
    row_array = np.concatenate((row_array,get_en_passant_channel(board)))
    
    row_array = np.concatenate((row_array,get_control_channel(board,1)))
    row_array = np.concatenate((row_array,get_control_channel(board,0)))
    
    #king_safety_channel_white = get_king_safety_channel(board,1)
    #king_safety_channel_black = get_king_safety_channel(board,0)
    
    #pawn_structure_channel_white = get_pawn_structure_channel(board,1)
    #pawn_structure_channel_black = get_pawn_structure_channel(board,0)
    
    #attack_channel_white = get_attack_channel(board,1)
    #attack_channel_black = get_attack_channel(board,0)
    
    #pin_channel_white = get_pin_channel(board,1)
    #pin_channel_black = get_pin_channel(board,0)
    
    return row_array

def get_material_value(piece):
    mat = {
        1 : 100,  #pawn
        2 : 320,  #knight
        3 : 330,  #bishop
        4 : 500,  #rook
        5 : 950,  #queen
        6 : 0   #king
        }
    return mat[piece]

def get_score(board,engine):
    info = engine.analyse(board, chess.engine.Limit(time=0.1))
    score = info['score'].relative.score()
    if score == None:
        return None
    if board.turn == 0:
        return -score
    return score

def get_material_imbalance(board):
    wp = len(board.pieces(chess.PAWN, chess.WHITE))
    bp = len(board.pieces(chess.PAWN, chess.BLACK))

    wn = len(board.pieces(chess.KNIGHT, chess.WHITE))
    bn = len(board.pieces(chess.KNIGHT, chess.BLACK))

    wb = len(board.pieces(chess.BISHOP, chess.WHITE))
    bb = len(board.pieces(chess.BISHOP, chess.BLACK))

    wr = len(board.pieces(chess.ROOK, chess.WHITE))
    br = len(board.pieces(chess.ROOK, chess.BLACK))

    wq = len(board.pieces(chess.QUEEN, chess.WHITE))
    bq = len(board.pieces(chess.QUEEN, chess.BLACK))

    value = (
        get_material_value(1) * (wp - bp) +
        get_material_value(2) * (wn - bn) +
        get_material_value(3) * (wb - bb) +
        get_material_value(4) * (wr - br) +
        get_material_value(5) * (wq - bq)
    )
    return value
    
def get_piece_channel(board,piece_type,col):
    piece_board = [0 for x in range(64)] 
    
    dic = board.piece_map()
    
    for square in dic:
        piece = dic[square]
        if (piece.piece_type == piece_type) and (piece.color == col):
                piece_board[square] = 1
            
    return np.array(piece_board)
    
def get_turn_to_move_channel(board):
    if board.turn:
        return np.ones(64)
    else:
        return np.zeros(64)
    
def get_castling_channel(board,side,color):
    channel = np.zeros(64)
    if side == 'kingside':
        channel += board.has_kingside_castling_rights(color)
    else:
        channel += board.has_queenside_castling_rights(color)
    return channel

def get_en_passant_channel(board):
    channel = np.zeros(64)
    if board.has_legal_en_passant():
        channel[board.ep_square] = 1
    return channel

def get_control_channel(board,turn):
    control_channel = [0 for x in range(64)]
    temp_board = board.copy()
    
    importance_board = np.array([
        [1,1,1,1,1,1,1,1],
        [1,2,2,2,2,2,2,1],
        [1,2,3,3,3,3,2,1],
        [1,2,3,4,4,3,2,1],
        [1,2,3,4,4,3,2,1],
        [1,2,3,3,3,3,2,1],
        [1,2,2,2,2,2,2,1],
        [1,1,1,1,1,1,1,1],
        ])
    
    temp_board.turn = turn
    for x in temp_board.legal_moves:
        piece = temp_board.piece_at(x.from_square).piece_type
        control_channel[x.to_square] += get_material_value(piece)

    return np.array(control_channel)*importance_board.flatten()