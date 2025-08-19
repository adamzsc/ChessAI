import chess
import chess.engine
import chess.pgn

import pandas as pd
import numpy as np

import board_utils as b

def df_init(n_channels):
    try:
        df = pd.read_csv("data.csv")
    except:
              
        cols = [f'board_feature_{i}' for i in range(n_channels * 64)] + ['material_imbalance','evaluation']
        df = pd.DataFrame(columns = cols)

    return df

def analyse_game(df,game,engine):
    
    board = chess.Board()
    
    for move in game.mainline_moves():
        
        score = b.get_score(board,engine)

        if score != None:
            idx = df.shape[0]           
            
            row_array = b.get_board_channels(board)
            
            row_array = np.append(row_array,b.get_material_imbalance(board))
            row_array = np.append(row_array,score)
            
            df.loc[idx] = row_array

        board.push(move)
    
    return df

def analyse_PGN(pgn, engine, df):
        
    while True:
        game= chess.pgn.read_game(pgn)
        if game is not None:
            df = analyse_game(df,game,engine)
            
            print("--GAME--")
            
        else: 
            break  

    return df    


engine = chess.engine.SimpleEngine.popen_uci(r"C:\Users\amsmi\Downloads\stockfish\stockfish-windows-x86-64-avx2.exe")

#pgn_file = "lichess_elite_2013-09"
pgn_file = "lichess_elite_2015-01"

pgn = open(f"pgn_database/{pgn_file}.pgn")

df = df_init(20)

new_df = analyse_PGN(pgn, engine, df.copy())
new_df.to_csv("data.csv",index=False)
