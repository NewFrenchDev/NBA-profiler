import os
import gc

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.preprocessing import  MinMaxScaler
from sklearn.metrics import confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

from constants import *

#Don't use GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

SAMPLE_SIZE = 1000

class Predictions:

    def __init__(self, name): 
        self.name = name
        self.dataset = None
        self.features = []
        self.model_loaded = None
        self.win_rate = 0

    def load_dataset(self):
        self.dataset = pd.read_csv(TEAMS_BOXSCORE_PATH, index_col="Unnamed: 0", dtype=DATAFRAME_COLUMNS_TYPE)

    def import_model(self, model_selected):
        #Get the model selected
        model_file = MODELS_FILES.get(model_selected)

        #import model
        if model_selected == "Artificial Neural Network":
            self.model_loaded = load_model(os.path.join(ROOT, "models", model_file))
            
        else: 
            with open(os.path.join(ROOT, "models", model_file), 'rb') as f:
                self.model_loaded = pickle.load(f)

    def process_input_before_prediction(self, detailed_mode, *args):
        team_parameters = None
        opponent_parameters = None

        if detailed_mode:
            team_parameters = self.calculate_four_factors(*args[0])
            opponent_parameters = self.calculate_four_factors(*args[1])

        else:
            team_parameters = args[0]
            opponent_parameters = args[1]

        #Save team and opponent parameters in same array
        self.features.clear()
        for param in team_parameters:
            self.features.append(param)
        for param in opponent_parameters:
            self.features.append(param)

        array = np.array(self.features).reshape(1, -1)

        return array

    def free_the_balloon(self):
        #Show balloon if the prediction is more than 90%
        if self.win_rate > 90:
            st.balloons()
    
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

        return [eFG_rate_normalized, TOV_rate_normalized, off_rebound_rate_normalized, free_throw_normalized]
    
    def predict(self, array):
        prediction = self.model_loaded.predict_proba(array)
        self.win_rate = np.round(prediction[0][1]*100,2)
        st.write(f"Team has **{self.win_rate}**% to win with these parameters")

    def compare_result_to_prediction(self, model_selected, dataset_sampled, predictions):
        st.write("---")
        st.header("Analysis")
        row2_1, row2_space2 , row2_2 = st.beta_columns([1, 0.5, 2])

        number_good_prediction = 0

        for result, prediction in zip(dataset_sampled.loc[:, 'W/L'], predictions):
            if result == "L" and prediction[1] < 0.5:
                number_good_prediction += 1
            if result == 'W' and prediction[1] > 0.5:
                number_good_prediction += 1

        dataset_sampled["Win"] = [0 if result == 'L' else 1 for result in dataset_sampled["W/L"]]
        predictions_rounded = np.argmax(predictions, axis=-1)

        #Figures

        #Confusion matrix
        cm = confusion_matrix( dataset_sampled["Win"] , predictions_rounded)
        fig = plt.figure(figsize=(5,5))
        ax = sns.heatmap(cm, annot=True,  fmt="d")
        ax.set_title(model_selected)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Real')
        ax.set_aspect('auto')

        fig_pie = px.pie(values=[number_good_prediction, SAMPLE_SIZE - number_good_prediction],
                         names=["Good predictions", "Bad predictions"],
                         title="Prediction accuracy")
        
        with row2_1:
            st.write('Confusion matrix ')
            st.write(fig)
            # st.markdown("""
            # The number of 
            # """)

        with row2_2:
            st.plotly_chart(fig_pie, use_container_width=True)

            del fig_pie

        self.win_rate = np.round(number_good_prediction/SAMPLE_SIZE, 2) * 100


    def test_prediction_on_dataset_sample(self, model_selected, dataset_sampled):
        columns = ["Effective Field Goal Percentage", "Turnover percentage",
                   "Offensive rebounding percentage", 'Percent of Points (Free Throws)',
                   "Opponent Effective Field Goal Percentage", "Opponent Turnover percentage",
                   "Opponent Offensive rebounding percentage", "Opponent Percent of Points (Free Throws)"]

        features = dataset_sampled.loc[:, columns]
        features = np.array(features)

        scaling = MinMaxScaler(feature_range=(0,1))
        scaled_features = scaling.fit_transform(features)

        predictions = self.model_loaded.predict_proba(scaled_features)

        dataframe_predictions = pd.DataFrame(predictions, columns=['L', 'W'])

       
        row1_1, row1_space, row1_2 = st.beta_columns([2, 1, 2])
        with row1_1:
            st.header("Prediction/Classification")
            st.write(dataframe_predictions)
            
        with row1_2:
            st.header("Results of the match")
            st.write( dataset_sampled.loc[:, 'W/L'].reset_index())

        self.compare_result_to_prediction(model_selected, dataset_sampled, predictions)

        gc.collect()


    def display_code(self, model_selected):
        if model_selected == 'Artificial Neural Network':
            st.markdown("""
            This Artificial Neural Network has been created with the API [Keras](https://keras.io/api/) from [Tensorflow](https://www.tensorflow.org/api_docs/python/tf).\n  
            """)
        st.code(CODE_PER_MODEL.get(model_selected))


    def display_prediction_view(self, prediction_option, detailed_mode, model_selected, *args):

        self.load_dataset()

        st.markdown("If you observe some balloons on your screen... It means the prediction is **above 90**%! 🔥🔥")

        self.import_model(model_selected)

        #Calculate the four factor if detailed mode selected
        if prediction_option == "Predict a match":
            features = self.process_input_before_prediction(detailed_mode, *args)
            st.header(f"Personal prevision test with {model_selected}")
            self.predict(features)

        #Test model on a sample of the dataset
        elif prediction_option == "Test models":
            
            st.write(f"**{model_selected}** model is used for this prediction")

            st.header('About the model')
            self.display_code(model_selected)

            st.header("Random sample from the dataset for the test prediction")
            st.markdown("This sample is chosen randomly each time the app re-run")
            sample = self.dataset.sample(SAMPLE_SIZE)
            st.dataframe(sample)

            self.test_prediction_on_dataset_sample(model_selected ,sample)

        self.free_the_balloon()

        #Always launch the garbage collector to free the memory
        gc.collect()