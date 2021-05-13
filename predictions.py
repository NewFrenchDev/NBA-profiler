import os
import shutil
import pathlib
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from joblib import load, dump

from constants import *


class Predictions:

    def __init__(self, name): 
        self.name = name
        self.path_boxscore = TEAMS_BOXSCORE_PATH
        self.boxscore = None
        self.features = None
        self.model = None

    def initiate_dashboard(self, model_selected, *arg):

        st.write(*arg[0])

        model_file = MODELS_FILES.get(model_selected)

        st.write(model_file)
        #import model
        # self.model = load(model_selected)

        # self.features = np.array[args]
        pass


    def predict_result(self):
        numeric_features = [i for i in range(10)] # Positions of numeric columns in X_train/X_test
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])

        features_scaled = preprocessor.fit_transform(self.features)

        prediction =  self.model.predict(features_scaled)

    def dashboard_first_row(self):
        pass

    def dashboard_second_row(self):
        pass
