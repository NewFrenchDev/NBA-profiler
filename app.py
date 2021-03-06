import tracemalloc
import gc
import shutil
from numpy.lib.npyio import load

import streamlit as st 
import pandas as pd


from constants import *
from predictions import Predictions
from data_analysis import DataAnalysisBoard

# SETUP ------------------------------------------------------------------------
st.set_page_config(page_title='Data Science in NBA',
                   layout="wide")

# tracemalloc.start()

def uncompress_file():
    shutil.unpack_archive(FILE_GZ_TO_DECOMPRESS, TEAMS_DATA_PATH)


# Initialisation of each view --------------------------------------------------
# Optimisation for Heroku and its 512ram and low memory
# @st.cache(allow_output_mutation=True)
def create_prediction_view(name):
    predictions_view = Predictions(name)
    return predictions_view

# @st.cache(allow_output_mutation=True)
def create_data_analysis_board(name):
    analysis_board = DataAnalysisBoard(name)
    return analysis_board

def load_csv(filepath, dtype):
    dataset = pd.read_csv(filepath, index_col="Unnamed: 0", dtype=dtype)
    return dataset

# The app
def setup():

    #Check file 
    if not os.path.isfile(TEAMS_BOXSCORE_PATH):
        uncompress_file()

    #Variable for stocking values and send it to prediction view
    team_parameters = []
    opponent_parameters = []

    dataset = load_csv(TEAMS_BOXSCORE_PATH, DATAFRAME_COLUMNS_TYPE)

    data_analysis_board = create_data_analysis_board(name='DataBoard')
    predictions = create_prediction_view(name='Prediction')
    
    #Sidebar

    st.sidebar.header('Playground')
    select_display = st.sidebar.radio('Sections', ['Data vizualisation','Match prediction'])

    #Main Page

    # ROW 1 ------------------------------------------------------------------------

    row1_1, row1_space, row1_2 = st.beta_columns([3, 1, 2])

    row1_1.title('NBA Win/Loss Classifier 🏀')

    with row1_2:
        st.write('')
        row1_2.subheader(
        'A Web App by [Gérard LEMOING](https://www.linkedin.com/in/gérard-lemoing-807099138/)')

    if select_display == 'Data vizualisation':

        #Team to visualize
        teams = dataset["Team"].unique().tolist()
        with st.sidebar.form(key='submit_teams'):
            teams_selected = st.multiselect('Teams', options=teams, default=['LAL', 'GSW', 'IND', 'BOS'])
            st.form_submit_button('Update')

        #Element to show for density 
        columns_from_dataset = dataset.columns.tolist()[5:]
        with st.sidebar.form(key='submit_group'):
            density_distribution_group = st.multiselect('Density distribution group', options=columns_from_dataset,
                                                        default=["Points", "3 Points Made", "Effective Field Goal Percentage", "Turnover"],)
            st.form_submit_button("Show distribution plot 🔥")

        show_four_factors = st.sidebar.checkbox('Show Four Factors')

        data_analysis_board.display_view(teams_selected, density_distribution_group, show_four_factors)

    elif select_display == 'Match prediction':

        #update sidebar
        list_of_model = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting',
                         'K Nearest Neighbors', 'AdaBoost', 'Artificial Neural Network']
        model_selected = st.sidebar.selectbox('Select a model: ', options=list_of_model)

        prediction_option_selected = st.sidebar.radio('Prediction options', ['Model\'s information', 'Predict a match'])

        if prediction_option_selected == 'Predict a match':
            detailed_mode = st.sidebar.checkbox('Detailed mode')
        else:
            detailed_mode = False

        #parameters
        if prediction_option_selected == 'Predict a match' and detailed_mode:
            
            st.sidebar.header('Team')
            field_goal_made = st.sidebar.slider('Field goal made', min_value=0, max_value=100, value=25, key='0')
            field_goal_attempt = st.sidebar.slider('Field goal attempt', min_value=1, max_value=100, value=25, key='1')
            three_pt_made = st.sidebar.slider('3 Point made', min_value=0, max_value=20, value=5, key='2')
            off_rebound = st.sidebar.slider('Offensive rebound', min_value=1, max_value=40, value=10, key='3')
            opp_def_rebound = st.sidebar.slider('Opponent offensive rebound', min_value=1, max_value=40, value=10, key='4') 
            turnover = st.sidebar.slider('Turnover', min_value=0, max_value=30, value=5, key='5')
            free_throw_made = st.sidebar.slider('Free Throw made', min_value=0, max_value=20, value=5, key='6')
            free_throw_attempted = st.sidebar.slider('Free Throw attempt', min_value=0, max_value=30, value=10, key='7')

            st.sidebar.header("Team's opponent")
            opp_field_goal_made = st.sidebar.slider('Field goal made', min_value=0, max_value=100, value=25)
            opp_field_goal_attempt = st.sidebar.slider('Field goal attempt', min_value=1, max_value=100, value=25)
            opp_three_pt_made = st.sidebar.slider('3 Point made', min_value=0, max_value=20, value=5)
            opp_off_rebound = st.sidebar.slider('Offensive rebound', min_value=1, max_value=40, value=10)
            opp_def_rebound_opp = st.sidebar.slider('Opponent offensive rebound', min_value=1, max_value=40, value=10) 
            opp_turnover = st.sidebar.slider('Turnover', min_value=0, max_value=30, value=5)
            opp_free_throw_made = st.sidebar.slider('Free Throw made', min_value=0, max_value=20, value=5)
            opp_free_throw_attempt = st.sidebar.slider('Free Throw attempt', min_value=0, max_value=30, value=10)

            #Save paramaters in a list to send to the the prediction class
            team_parameters = [field_goal_made, field_goal_attempt, three_pt_made, off_rebound,
                            opp_def_rebound, turnover, free_throw_made, free_throw_attempted]

            opponent_parameters = [opp_field_goal_made, opp_field_goal_attempt, opp_three_pt_made, opp_off_rebound,
                                opp_def_rebound_opp, opp_turnover, opp_free_throw_made, opp_free_throw_attempt]

        elif prediction_option_selected == 'Predict a match' and not detailed_mode:

            st.sidebar.header('Team')

            efg_rate = st.sidebar.slider('Effective Field Goal Percentage', min_value=0, max_value=100, value=25, key='8') /100
            tov_rate = st.sidebar.slider('Turnover percentage', min_value=0, max_value=100, value=25, key='9') / 100
            off_rebound_rate = st.sidebar.slider('Offensive rebounding percentage', min_value=0, max_value=100, value=25, key='10') / 100
            free_throw_rate = st.sidebar.slider('Percent of Free Throw Made', min_value=0, max_value=100, value=25, key='11') / 100

            st.sidebar.header("Team's opponent")

            opp_efg_rate = st.sidebar.slider('Effective Field Goal Percentage', min_value=0, max_value=100, value=25) / 100
            opp_tov_rate = st.sidebar.slider('Turnover percentage', min_value=0, max_value=100, value=25) / 100
            opp_off_rebound_rate = st.sidebar.slider('Offensive rebounding percentage', min_value=0, max_value=100, value=25) / 100
            opp_free_throw_rate = st.sidebar.slider('Percent of Free Throw Made', min_value=0, max_value=100, value=25) / 100

            #Save paramaters in a list to send to the the prediction class
            team_parameters = [efg_rate, tov_rate, off_rebound_rate, free_throw_rate]
            opponent_parameters = [opp_efg_rate, opp_tov_rate, opp_off_rebound_rate, opp_free_throw_rate]

        
        predictions.display_prediction_view(prediction_option_selected, detailed_mode, model_selected, team_parameters, opponent_parameters)
    
 
    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics('lineno')

    # print('--------------TOP 10-----------')
    # for stat in top_stats[:10]:
    #     print(stat)

if __name__ == '__main__':

    #Launch the app
    setup()

    #Always launch the garbage collector to free the memory
    gc.collect()