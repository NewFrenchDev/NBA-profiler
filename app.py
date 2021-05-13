import os
import base64

import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import psycopg2
# from sqlalchemy import create_engine
from dotenv import load_dotenv
import tracemalloc
import gc

from predictions import Predictions
from data_analysis import DataAnalysisBoard

load_dotenv('.env')

# tracemalloc.start()

# SETUP ------------------------------------------------------------------------
st.set_page_config(page_title='NBA and big data',
                   layout="wide")

# Initialisation of each view --------------------------------------------------
# Optimisation for Heroku and its 512ram and low memory
# @st.cache(allow_output_mutation=True)
def create_prediction_view(name='Test'):
    predictions_view = Predictions(name)
    return predictions_view

# @st.cache(allow_output_mutation=True)
def create_data_analysis_board(name='Test'):
    analysis_board = DataAnalysisBoard(name)
    return analysis_board

# The app
def setup():

    predictions = create_prediction_view(name='Luffy')
    data_analysis_board = create_data_analysis_board(name='Zoro')

    #Sidebar

    st.sidebar.header('User Input Features')
    select_display = st.sidebar.radio('Cool Features', ['Data visualisation','Match prediction'])

    #Main Page

    # ROW 1 ------------------------------------------------------------------------

    row1_1, row1_space, row1_2 = st.beta_columns(
        (2, 2, 2)
        )

    row1_1.title('NBA data analysis')

    with row1_2:
        st.write('')
        row1_2.subheader(
        'A Web App by [Gérard LEMOING]')

    if select_display == 'Match prediction':

        #update sidebar
        list_of_model = ['Regression logistic', 'Decision Tree', 'Random Forest', 'XGBoost',
                         'K Nearest Neighbors', 'AdaBoost', 'Artificial Neural Network']
        model_selected = st.sidebar.selectbox('Select a model: ', options=list_of_model)

        #parameters
        st.sidebar.header('Team')
        field_goal_made = st.sidebar.slider('Field goal made', min_value=0, max_value=100, value=25, key='0')
        field_goal_attempted = st.sidebar.slider('Field goal attempted', min_value=0, max_value=100, value=25, key='1')
        three_pt_made = st.sidebar.slider('3 Point made', min_value=0, max_value=100, value=25, key='2')
        off_rebound = st.sidebar.slider('Offensive rebound', min_value=0, max_value=100, value=25, key='3')
        opp_def_rebound = st.sidebar.slider('Opponent offensive rebound', min_value=0, max_value=100, value=25, key='4') 
        turnover = st.sidebar.slider('Turnover', min_value=0, max_value=100, value=25, key='5')
        free_throw_made = st.sidebar.slider('Free Throw made', min_value=0, max_value=100, value=25, key='6')
        free_throw_attempted = st.sidebar.slider('Free Throw attempted', min_value=0, max_value=100, value=25, key='7')

        team_parameters = [field_goal_made, field_goal_attempted, three_pt_made, off_rebound,
                           opp_def_rebound, turnover, free_throw_made, free_throw_attempted]

        st.sidebar.header("Team's opponent")
        opp_field_goal_made = st.sidebar.slider('Field goal made', min_value=0, max_value=100, value=25)
        opp_field_goal_attempted = st.sidebar.slider('Field goal attempted', min_value=0, max_value=100, value=25)
        opp_three_pt_made = st.sidebar.slider('3 Point made', min_value=0, max_value=100, value=25)
        opp_off_rebound = st.sidebar.slider('Offensive rebound', min_value=0, max_value=100, value=25)
        opp_def_rebound_opp = st.sidebar.slider('Opponent offensive rebound', min_value=0, max_value=100, value=25) 
        opp_turnover = st.sidebar.slider('Turnover', min_value=0, max_value=100, value=25)
        opp_free_throw_made = st.sidebar.slider('Free Throw made', min_value=0, max_value=100, value=25)
        opp_free_throw_attempted = st.sidebar.slider('Free Throw attempted', min_value=0, max_value=100, value=25)

        opponent_parameters = [opp_field_goal_made, opp_field_goal_attempted, opp_three_pt_made, opp_off_rebound,
                               opp_def_rebound_opp, opp_turnover, opp_free_throw_made, opp_free_throw_attempted]

        predictions.initiate_dashboard(model_selected, team_parameters, opponent_parameters)
        predictions.dashboard_first_row()
        predictions.dashboard_second_row()
    elif select_display == 'Data visualisation':
        data_analysis_board.display_view()
 
    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics('lineno')

    # print('--------------TOP 10-----------')
    # for stat in top_stats[:10]:
    #     print(stat)

if __name__ == '__main__':

    setup()
    gc.collect()