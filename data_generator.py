import chess
import chess.engine
import chess.pgn

import pandas as pd
import numpy as np

import board_utils as b
import matplotlib.pyplot as plt

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

def plot_bar(arr,max=100):
    labels = ['<1000','1000-1200','1200-1400','1400-1600','1600-1800','1800-2000','>2000']
    plt.ylim(0,100)
    plt.ylabel('Games')
    plt.xlabel('Avg. Elo')
    plt.xticks(rotation=45)
    plt.bar(labels,arr)
    plt.show()

def analyse_PGN_file(pgn, engine, df):
    
    games_analysed = np.array([0,0,0,0,0,0,0])
        
    while True:
        if np.sum(games_analysed > 100) == len(games_analysed):
            break
        
        game = chess.pgn.read_game(pgn)
        
        if game is None:
            break
            
        else:  
        
            white_elo = game.headers.get("WhiteElo")
            black_elo = game.headers.get("BlackElo")
            
            
            if white_elo.isdigit() or black_elo.isdigit():
                if white_elo.isdigit() and black_elo.isdigit():
                    avg_elo = (int(white_elo) + int(black_elo)) / 2
                elif white_elo.isdigit():
                    avg_elo = int(white_elo)
                else:
                    avg_elo = int(black_elo)
                    
                if avg_elo < 1000:
                    digit = 0
                elif avg_elo > 2000:
                    digit = 6
                else:
                    digit = int(1 + (avg_elo - 1000) // 200)
                
                if games_analysed[digit] == 100:
                    continue
                else:
                    games_analysed[digit] += 1
                    plot_bar(games_analysed)
            
            df = analyse_game(df,game,engine)
            
            print("--GAME--")
        
    return df    


engine = chess.engine.SimpleEngine.popen_uci(r"C:\Users\amsmi\Downloads\stockfish\stockfish-windows-x86-64-avx2.exe")

pgn_file_name = "pgn_database"

pgn_file = open(f"{pgn_file_name}.pgn")

df = df_init(20)

new_df = analyse_PGN_file(pgn_file, engine, df.copy())
new_df.to_csv("data.csv",index=False)
