import os

ROOT = os.path.dirname(__file__)
DATE_FORMAT = '%m/%d/%Y'
PLAYERS_DATA_PATH = os.path.join(ROOT, 'data', 'players') 
PLAYER_BOXSCORE_PATH = os.path.join(ROOT, PLAYERS_DATA_PATH, 'players-boxscores.csv') 
PLAYER_BOXSCORE_ADVANCED_PATH = os.path.join(ROOT, PLAYERS_DATA_PATH, 'players-boxscores-advanced.csv') 
PLAYER_BOXSCORE_SCORING_PATH = os.path.join(ROOT, PLAYERS_DATA_PATH, 'players-boxscores-scoring.csv') 
PLAYER_BOXSCORE_TRADITIONAL_PATH = os.path.join(ROOT, PLAYERS_DATA_PATH, 'players-boxscores-traditional.csv') 
PLAYER_BOXSCORE_USAGE_PATH = os.path.join(ROOT, PLAYERS_DATA_PATH, 'players-boxscores-usage.csv') 
FILE_GZ_TO_DECOMPRESS = os.path.join(ROOT, PLAYERS_DATA_PATH, 'players.tar.gz') 