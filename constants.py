import os

ROOT = os.path.dirname(__file__)
DATE_FORMAT = '%m/%d/%Y'
TEAMS_DATA_PATH = os.path.join(ROOT, 'data', 'teams') 
TEAMS_BOXSCORE_PATH = os.path.join(ROOT, TEAMS_DATA_PATH, 'teams-simplified-boxscore.csv') 
FILE_GZ_TO_DECOMPRESS = os.path.join(ROOT, TEAMS_DATA_PATH, 'teams-simplified-boxscore.tar.gz') 
MODELS_FILES = {
    'Logistic Regression': 'logistic_regression_model',
    'Decision Tree': "decision_tree_model",
    'Random Forest': "random_forest_model",
    'XGBoost': "gradient_boosting_model",
    'K Nearest Neighbors': "knn_model",
    'AdaBoost': "adaboost_model",
    'Artificial Neural Network': "artificial_neural_network_model"
}
DATAFRAME_COLUMNS_TYPE = {
    'Points': 'int16',
    'Field Goal Made': 'int16',	
    'Field Goal Attempt': 'int16',	
    'Percent of Field Goal Made': 'float16',
    '3 Points Made':  'int8',
    '3 Points Attempt': 'int8',	
    'Percent of 3 Points Made': 'float16',
    'Free Throw Made': 'int8',
    'Free Throw Attempt': 'int8',
    'Percent of Free Throw Made': 'float16',
    'Offensive Rebounding': 'int8',
    'Defensive Rebounding': 'int8',	
    'Rebounding': 'int8',	
    'Assist': 'int8',	
    'Steal': 'int8',
    'Block': 'int8',
    'Turnover': 'int8',
    'Personal Fouls': 'int8',
    'ASTRatio': 'float16',	
    'Offensive rebounding percentage': 'float16',
    'Defensive rebounding percentage': 'float16',
    'Rebounding percentage': 'float16',	
    'Turnover percentage': 'float16',
    'Effective Field Goal Percentage': 'float16',
    'True Shooting Percentage': 'float16',	
    'Player Impact Estimate': 'float16',
    'Percent of Points (2-Point Field Goals)': 'float16',
    'Percent of Points (2-Point Field Goals: Mid Range)': 'float16',
    'Percent of Points (3-Point Field Goals)': 'float16',
    'Percent of Points (Free Throws)': 'float16',
    'Year': 'int16',	
    'Opponent Effective Field Goal Percentage': 'float16',
    'Opponent Turnover percentage': 'float16',
    'Opponent Offensive rebounding percentage': 'float16',
    'Opponent Percent of Points (Free Throws)': 'float16'
}