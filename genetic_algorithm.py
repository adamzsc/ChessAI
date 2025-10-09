import chess
import chess.engine
import engine
import numpy as np
import torch

import board_utils as b
import joblib

import asyncio
import sys

import copy
from multiprocessing import Pool, cpu_count
import time
 

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
# --- GA Configuration ---
POPULATION_SIZE = 20
NUM_GENERATIONS = 10
MUTATION_RATE = 0.05
TOURNAMENT_ROUNDS = 5 # Each model plays this many games (White + Black against SF)
GAME_DEPTH = 2 # Fixed depth for CNN model search during fitness evaluation
MAX_GAME_MOVES = 80 # Maximum moves per game before drawing

# --- Stockfish Configuration ---

STOCKFISH_ELO = 1400 # Slightly increased ELO for better positional challenge
STOCKFISH_TIME_LIMIT = 0.05 # Time in seconds per move for Stockfish

class GAEngine(engine.Engine):
    
    def __init__(self,model_state_dict,use_opening_table = False,max_depth=20,time_limit=10):
        
        scalers_board_filename = 'scalers_board.joblib'
        scaler_extra_filename = 'scaler_extra.joblib'
        scaler_y_filename = 'scaler_y.joblib'
        
        self.scalers_board = joblib.load("scalers/"+scalers_board_filename)
        self.scaler_extra = joblib.load("scalers/"+scaler_extra_filename)
        self.scaler_y = joblib.load("scalers/"+scaler_y_filename)
        
        self.use_opening_table = use_opening_table
        
        self.n_input_channels = 20
        self.extra_features_size = 1         
        
        self.model = b.CNN_Regressor(self.n_input_channels, self.extra_features_size)
        self.model.load_state_dict(model_state_dict)
        self.model.eval() 
        
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
        

def calculate_fitness_score(board: chess.Board, color: chess.Color) -> float:
    """
    Calculates a heuristic score based on early development and center control 
    for the given color (White or Black). This score is used as fitness.
    """
    score = np.random.randint(0,5)

    return score

def initialize_population(model_path):
    """
    Loads the base model and creates copies (individuals) for the initial population.
    The scalers are now loaded by GAEngine, so they are not stored here.
    """
    base_state_dict = b.load_model(b.CNN_Regressor, model_path, 20, 1).state_dict()
    
    population = []
    for _ in range(POPULATION_SIZE):
        individual_state = copy.deepcopy(base_state_dict)
        population.append({'model_state': individual_state, 'fitness': 0.0}) 
    
    return population

def mutate_model_weights(state_dict, rate, strength=0.01):
    """Applies random mutation to the weights of the model state dictionary."""
    new_state_dict = copy.deepcopy(state_dict)
    with torch.no_grad():
        for name, param in new_state_dict.items():
            if 'weight' in name and len(param.shape) > 1: 
                mask = torch.rand(param.shape) < rate
                noise = torch.randn(param.shape) * strength
                param[mask] += noise[mask].to(param.device)
    return new_state_dict

def crossover_models(parent1_state, parent2_state):
    """Performs uniform crossover on model weights."""
    child_state = copy.deepcopy(parent1_state)
    for name, param1 in parent1_state.items():
        param2 = parent2_state[name]
        if isinstance(param1, torch.Tensor) and param1.shape == param2.shape:
            mask = torch.rand(param1.shape) < 0.5 
            child_param = param1.clone()
            child_param[mask] = param2[mask]
            child_state[name] = child_param
    return child_state

def play_game(config):
    """
    Simulates a single game between the CNN model and Stockfish.
    The fitness is the total accumulated positional score for the CNN model.
    Returns: A tuple (model_index, total_positional_score)
    """
    # Removed scalers from the config tuple unpacking
    
    (model_state, model_index, max_moves, game_depth, is_model_white, sf_elo, sf_limit) = config
    
    # Initialize CNN Engine (now loads scalers internally)
    try:
        cnn_engine = GAEngine(model_state)
        sf_engine = chess.engine.SimpleEngine.popen_uci(r"C:\Users\Adam\Desktop\Python\Chess\other\stockfish\stockfish-windows-x86-64-avx2.exe") 
        sf_engine.configure({
            "UCI_LimitStrength": True,
            "UCI_Elo": sf_elo
        })
    except Exception as e:
        # Cleanly shut down engine if it started, before exiting
        try: sf_engine.quit() 
        except: pass
        print(f"Error initializing engines for model {model_index}: {e}")
        return model_index, 0.0 # Return zero score on failure
        
    board = chess.Board()
    moves_played = 0
    model_color = chess.WHITE if is_model_white else chess.BLACK
    
    # Accumulator for the positional fitness score
    total_fitness_score = 0.0 
    
    try:
        while not board.is_game_over(claim_draw=True) and moves_played < max_moves:
            
            if board.turn == model_color:
                # --- CNN Model's Turn: Calculate Move and Record Fitness ---
                
                # 1. Calculate Positional Score of the current board state
                current_fitness_score = calculate_fitness_score(board, model_color)
                total_fitness_score += current_fitness_score
                
                # 2. Determine best move using the CNN engine
                move = cnn_engine.find_best_move(board.copy())
            else:
                # Stockfish's turn
                result = sf_engine.play(board, chess.engine.Limit(time=sf_limit))
                move = result.move
            
            if move is None: 
                break 
                
            board.push(move)
            moves_played += 1
                
        # Return the accumulated positional score as fitness
        return model_index, total_fitness_score
    
    except Exception as e:
        print(f"Game error for model {model_index}: {e}")
        return model_index, 0.0 # Return zero score on error
        
    finally:
        sf_engine.quit()


def run_tournament_parallel(population, num_rounds, depth, max_moves=MAX_GAME_MOVES):
    """
    Runs a parallel tournament where each model plays Stockfish and fitness 
    is based on the accumulated positional score.
    Removed 'scalers' from function signature.
    """

    game_configs = []
    
    for i, individual in enumerate(population):
        for r in range(num_rounds):
            
            # Game 1: CNN Model (White) vs Stockfish (Black)
            game_configs.append((
                individual['model_state'], 
                i, # Model Index 
                max_moves, 
                depth,
                True, # is_model_white
                STOCKFISH_ELO,
                STOCKFISH_TIME_LIMIT
            ))
            
            # Game 2: Stockfish (White) vs CNN Model (Black)
            game_configs.append((
                individual['model_state'], 
                i, # Model Index
                max_moves, 
                depth,
                False, # is_model_white
                STOCKFISH_ELO,
                STOCKFISH_TIME_LIMIT
            ))
            
    num_workers = cpu_count()
    print(f"Starting tournament with {len(game_configs)} total games using {num_workers} processes...")

    # Use Pool to run games in parallel
    with Pool(num_workers) as p:
        results = p.map(play_game, game_configs)

    # Aggregate results: sum up the positional scores
    model_scores = {i: [] for i in range(len(population))}
    for idx, score in results:
        model_scores[idx].append(score)

    # Calculate average fitness (Average Positional Score per game)
    new_population = []
    for i in range(len(population)):
        num_games = len(model_scores[i])
        total_score = sum(model_scores[i])
        avg_score = total_score / num_games if num_games > 0 else 0.0
        
        new_population.append({
            'model_state': population[i]['model_state'],
            'fitness': avg_score,
        })

    return new_population


class GeneticOptimizer:
    # Removed scalers_path from init argument
    def __init__(self, base_model_path, population_size=POPULATION_SIZE):
        
        # Scaler loading removed as GAEngine now handles it internally.
        
        self.base_model_path = base_model_path
        self.population_size = population_size
        self.population = initialize_population(base_model_path)
        
    def evolve(self, num_generations=NUM_GENERATIONS, tournament_rounds=TOURNAMENT_ROUNDS, game_depth=GAME_DEPTH):
        
        print("--- Starting Genetic Algorithm Evolution ---")
        
        for generation in range(num_generations):
            start_time = time.time()
            
            # 1. Evaluate Fitness (Parallel Tournament vs Stockfish)
            # Removed scalers argument from run_tournament_parallel call
            self.population = run_tournament_parallel(
                self.population, 
                tournament_rounds, 
                game_depth
            )
            
            # Sort population by fitness
            self.population.sort(key=lambda x: x['fitness'], reverse=True)
            
            best_fitness = self.population[0]['fitness']
            avg_fitness = np.mean([p['fitness'] for p in self.population])
            
            print(f"\nGeneration {generation+1}/{num_generations} complete. Time: {time.time() - start_time:.2f}s")
            print(f"  Best Fitness (Avg Positional Score): {best_fitness:.4f}")
            print(f"  Average Fitness: {avg_fitness:.4f}")

            # 2. Selection and Next Generation creation
            new_population = []
            
            # Elitism: Keep the best individual
            new_population.append(self.population[0]) 

            # Create the rest of the new generation
            while len(new_population) < self.population_size:
                parent1, parent2 = self._select_parents()
                
                # Crossover
                child_state = crossover_models(parent1['model_state'], parent2['model_state'])
                
                # Mutation
                child_state_mutated = mutate_model_weights(child_state, MUTATION_RATE)
                
                new_population.append({
                    'model_state': child_state_mutated, 
                    'fitness': 0.0, 
                })
                
            self.population = new_population

        # Final best model
        self.population.sort(key=lambda x: x['fitness'], reverse=True)
        final_best_model_state = self.population[0]['model_state']
        
        print("\n--- Genetic Algorithm Finished ---")
        return final_best_model_state
    
    def _select_parents(self):
        """Tournament selection from the fittest half."""
        parents_pool = self.population[:self.population_size // 2]
        if len(parents_pool) < 2:
            return self.population[0], self.population[0]
            
        p1_idx, p2_idx = np.random.choice(len(parents_pool), size=2, replace=False)
        return parents_pool[p1_idx], parents_pool[p2_idx]


if __name__ == '__main__':
    # DUMMY PATHS - REPLACE WITH YOUR ACTUAL FILE STRUCTURE
    MODEL_PATH = 'cnn_model_v0.pth' 

    try:
        # Note: GAEngine loads scalers internally, so we don't pass a scalers path
        ga = GeneticOptimizer(MODEL_PATH, POPULATION_SIZE) 
        final_model_state = ga.evolve()
        
        # Optionally save the best model weights
        torch.save(final_model_state, 'best_evolved_cnn_model_positional.pth')
        print("\nSaved best evolved model state to 'best_evolved_cnn_model_positional.pth'")
            
    except FileNotFoundError as e:
        print(f"\nCRITICAL ERROR: {e}")
        print("Please check the path to your files.")
    except Exception as e:
        print(f"\nAn error occurred during GA execution: {e}")
