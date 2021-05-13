import os
import gc

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from joblib import load, dump
import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

from constants import *

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class Predictions:

    def __init__(self, name): 
        self.name = name
        self.dataset = None
        self.features = []
        self.model_loaded = None

    def initiate_dashboard(self, model_selected, *args):

        self.dataset = pd.read_csv(TEAMS_BOXSCORE_PATH, index_col="Unnamed: 0", dtype=DATAFRAME_COLUMNS_TYPE)

        st.write(*args[0])
        team_parameters = self.calculate_four_factors(*args[0])
        
        st.write(*args[1])
        opponent_parameters = self.calculate_four_factors(*args[1])

        self.features.clear()
        for param in team_parameters:
            self.features.append(param)
        for param in opponent_parameters:
            self.features.append(param)

        st.write(self.features)
        array = np.array(self.features).reshape(1, -1)

        model_file = MODELS_FILES.get(model_selected)

        #import model
        if model_selected == "Artificial Neural Network":
            self.model_loaded = load_model(os.path.join(ROOT, "models", model_file))
            self.predict_ann(array)

    def normalize_data(self, df, param_name, param_value):
        max = df[param_name].max()
        min = df[param_name].min()
        value_normalized = (param_value - min) / (max - min)
        return value_normalized

    def calculate_four_factors(self, *arg):
        eFG_rate = ((arg[0] + 0.5 * arg[2]) / arg[1])*100
        eFG_rate_normalized = self.normalize_data(self.dataset, 'Effective Field Goal Percentage', eFG_rate)
        TOV_rate = (arg[5] / (arg[1] + 0.44 * arg[7] + arg[5]))*100
        TOV_rate_normalized = self.normalize_data(self.dataset, 'Turnover percentage', TOV_rate)
        off_rebound_rate = (arg[3] /(arg[3] + arg[4]))*100
        off_rebound_rate_normalized = self.normalize_data(self.dataset, 'Offensive rebounding percentage', off_rebound_rate)
        free_throw = (arg[6] / arg[7])*100
        free_throw_normalized =  self.normalize_data(self.dataset, 'Percent of Points (Free Throws)', free_throw)

        st.write(eFG_rate, TOV_rate, off_rebound_rate, free_throw)
        st.write(eFG_rate_normalized, TOV_rate_normalized, off_rebound_rate_normalized, free_throw_normalized)

        return [eFG_rate_normalized, TOV_rate_normalized, off_rebound_rate_normalized, free_throw_normalized]

    # def predict_result(self, array):
    #     numeric_features = [i for i in range(10)] # Positions of numeric columns in X_train/X_test
    #     numeric_transformer = Pipeline(steps=[
    #         ('scaler', StandardScaler())
    #     ])

    #     preprocessor = ColumnTransformer(
    #     transformers=[
    #         ('num', numeric_transformer, numeric_features)
    #     ])
    #     print(array)

    #     features_scaled = preprocessor.fit_transform(array)

    #     print(features_scaled)

    #     prediction =  self.model_loaded.predict(features_scaled)

    #     print(prediction)

    #     st.write(prediction)



    
    def predict_ann(self, array):

        prediction = self.model_loaded.predict(array)
        st.write(prediction)

        del prediction
        gc.collect()


    def dashboard_first_row(self):
        pass

    def dashboard_second_row(self):
        pass
