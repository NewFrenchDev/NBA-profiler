import streamlit as st 
import tracemalloc
import gc

from predictions import Predictions
from data_analysis import DataAnalysisBoard

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

    data_analysis_board = create_data_analysis_board(name='Zoro')
    predictions = create_prediction_view(name='Luffy')
    team_parameters = []
    opponent_parameters = []
    
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

    if select_display == 'Data visualisation':
        data_analysis_board.display_view()

    elif select_display == 'Match prediction':

        prediction_option_selected = st.sidebar.radio('Prediction options', ['Test models', 'Predict a match'])

        #update sidebar
        list_of_model = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost',
                         'K Nearest Neighbors', 'AdaBoost', 'Artificial Neural Network']
        model_selected = st.sidebar.selectbox('Select a model: ', options=list_of_model)

        detailed_mode = st.sidebar.checkbox('Detailed mode :')

        #parameters
        st.sidebar.header('Team')

        if detailed_mode:
            field_goal_made = st.sidebar.slider('Field goal made', min_value=0, max_value=100, value=25, key='0')
            field_goal_attempted = st.sidebar.slider('Field goal attempted', min_value=0, max_value=100, value=25, key='1')
            three_pt_made = st.sidebar.slider('3 Point made', min_value=0, max_value=20, value=5, key='2')
            off_rebound = st.sidebar.slider('Offensive rebound', min_value=0, max_value=40, value=10, key='3')
            opp_def_rebound = st.sidebar.slider('Opponent offensive rebound', min_value=0, max_value=40, value=10, key='4') 
            turnover = st.sidebar.slider('Turnover', min_value=0, max_value=30, value=5, key='5')
            free_throw_made = st.sidebar.slider('Free Throw made', min_value=0, max_value=20, value=5, key='6')
            free_throw_attempted = st.sidebar.slider('Free Throw attempted', min_value=0, max_value=30, value=10, key='7')

            team_parameters = [field_goal_made, field_goal_attempted, three_pt_made, off_rebound,
                            opp_def_rebound, turnover, free_throw_made, free_throw_attempted]
        else:
            eFG_rate = st.sidebar.slider('Effective Field Goal Percentage', min_value=0, max_value=100, value=25, key='8')
            TOV_rate = st.sidebar.slider('Turnover percentage', min_value=0, max_value=100, value=25, key='9')
            off_rebound_rate = st.sidebar.slider('Offensive rebounding percentage', min_value=0, max_value=100, value=25, key='10')
            free_throw_rate = st.sidebar.slider('Free Throw percentage', min_value=0, max_value=100, value=25, key='11')

        st.sidebar.header("Team's opponent")

        if detailed_mode:
            opp_field_goal_made = st.sidebar.slider('Field goal made', min_value=0, max_value=100, value=25)
            opp_field_goal_attempted = st.sidebar.slider('Field goal attempted', min_value=0, max_value=100, value=25)
            opp_three_pt_made = st.sidebar.slider('3 Point made', min_value=0, max_value=20, value=5)
            opp_off_rebound = st.sidebar.slider('Offensive rebound', min_value=0, max_value=40, value=10)
            opp_def_rebound_opp = st.sidebar.slider('Opponent offensive rebound', min_value=0, max_value=40, value=10) 
            opp_turnover = st.sidebar.slider('Turnover', min_value=0, max_value=30, value=5)
            opp_free_throw_made = st.sidebar.slider('Free Throw made', min_value=0, max_value=20, value=5)
            opp_free_throw_attempted = st.sidebar.slider('Free Throw attempted', min_value=0, max_value=30, value=10)

            opponent_parameters = [opp_field_goal_made, opp_field_goal_attempted, opp_three_pt_made, opp_off_rebound,
                                opp_def_rebound_opp, opp_turnover, opp_free_throw_made, opp_free_throw_attempted]
        else:
            eFG_rate = st.sidebar.slider('Effective Field Goal Percentage', min_value=0, max_value=100, value=25)
            TOV_rate = st.sidebar.slider('Turnover percentage', min_value=0, max_value=100, value=25)
            off_rebound_rate = st.sidebar.slider('Offensive rebounding percentage', min_value=0, max_value=100, value=25)
            free_throw_rate = st.sidebar.slider('Free Throw percentage', min_value=0, max_value=100, value=25)

        if detailed_mode:
            predictions.initiate_dashboard(model_selected, team_parameters, opponent_parameters)
    
 
    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics('lineno')

    # print('--------------TOP 10-----------')
    # for stat in top_stats[:10]:
    #     print(stat)

if __name__ == '__main__':

    setup()
    gc.collect()