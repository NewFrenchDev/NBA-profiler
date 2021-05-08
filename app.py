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

from dashboard import Dashboard
from data_analysis import DataAnalysisBoard

load_dotenv('.env')

tracemalloc.start()

# SETUP ------------------------------------------------------------------------
st.set_page_config(page_title='NBA Profiler Dashboard',
                   layout="wide")

# player_dashboard = Dashboard()

# def filedownload(dataframe):

#     csv = dataframe.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV file</a>'
#     return href
 

# @st.cache
# def load_data(year, team):

#     records = None
#     df=None

#     try:
#         cnx = create_engine(f"postgresql+psycopg2://{os.environ.get('USER')}:{os.environ.get('PASSWORD')}@{os.environ.get('HOST')}:{os.environ.get('PORT')}/{os.environ.get('NAME')}")
#         df = pd.read_sql_query(f"SELECT * FROM players_boxscore", con=cnx, parse_dates=['Game Date'])
#         df = df[df['Game Date'].dt.year == year]
#     except Exception as e:
#         print('Unable to connect to the database', e)

#     finally:
#         return df

@st.cache(allow_output_mutation=True)
def create_dashboard(name='Test'):
    dashboard = Dashboard(name)
    return dashboard

def create_data_analysis_board(name='Test'):
    analysis_board = DataAnalysisBoard(name)
    return analysis_board

def setup():

    player_dashboard = create_dashboard(name='Luffy')
    data_analysis_board = create_data_analysis_board(name='Zoro')

    #Sidebar

    st.sidebar.header('User Input Features')
    select_display = st.sidebar.radio('Cool Features', ['Raw data profiling','Dashboard', 'Track the ball'])

    #Main Page

    # ROW 1 ------------------------------------------------------------------------

    row1_1, row1_space, row1_2 = st.beta_columns(
        (2, 2, 2)
        )

    row1_1.title('NBA Profiler')

    with row1_2:
        st.write('')
        row1_2.subheader(
        'A Web App by [Gérard LEMOING]')

    if select_display == 'Dashboard':
        player_dashboard.initiate_dashboard()
        player_dashboard.dashboard_first_row()
        player_dashboard.dashboard_second_row()
    elif select_display == 'Raw data profiling':
        data_analysis_board.display()
    elif select_display == 'Track the ball':
        st.write('*Page in construction* 👷 ')
 
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    
    print("\n\n[Top 10]")
    for stat in top_stats[:10]:
        print(stat)

    # if select_display == 'Dashboard':

    #     selected_year = st.sidebar.selectbox('Year', list(reversed(range(2000, 2022))))

    #     selected_team = st.sidebar.selectbox('Team', ['WAS', 'TOR', 'LAL'])

    

    # if select_display == 'Dashboard':


    #     st.write("""

    #     # Dashboard

    #     """)


    #     if st.button('Connect to Database Postgresql'):

    #         test = load_data(selected_year, selected_team)
    #         test

    #         st.markdown(filedownload(test), unsafe_allow_html=True)


    #     # st.image('https://hoopdirt.com/wp-content/uploads/2018/10/ipad-court-690x340.png')

    #     #Upload dataset
    #     uploaded_file = st.file_uploader('Upload your input CSV file', type=["csv"])

    #     if uploaded_file is not None:
    #         dataframe_to_load = pd.read_csv(uploaded_file)
    #         st.dataframe(dataframe_to_load)


    #     col1, col2, col3 = st.beta_columns(3)

    #     col1.subheader('First section')

    #     col2.subheader('Second section')

    #     col3.subheader('Third section')



if __name__ == '__main__':

    setup()
    gc.collect()