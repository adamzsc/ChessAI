import chess
import lichess

import joblib
import torch
import time

import numpy as np

import board_utils as b
import engine
import requests

def user_go(board):
    while True:
        move_input = input("Enter Move: ")
        move = chess.Move.from_uci(move_input)
        if move in board.legal_moves:
            return move
        else:
            print("NOT A LEGAL MOVE")



if __name__ == '__main__':
    
    model_path = 'cnn_model_v0.pth'
         
    
    engine = engine.Engine(model_path)
    
    board = chess.Board()  
    user_col = 0
    
    while not board.is_game_over():
        if board.turn == user_col:
            move = user_go(board)
        else:
            move = engine.find_best_move(board.copy())
            
        print("\n"*10)
        print(('Black','White')[board.turn], "played", board.san(move))
        board.push(move)
        print(board)
        
    print("GAME END")