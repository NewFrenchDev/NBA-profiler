import os
import shutil
import pathlib
from datetime import datetime

import streamlit as st
import pandas as pd

from constants import *

class Dashboard:

    def __init__(self, name): 
        self.name = name
        self.path_boxscore = PLAYER_BOXSCORE_PATH
        self.boxscore = None
        self.unique_players = None
        self.player_selected = None
        self.number_of_game = 0

    def uncompress_file(self):
        shutil.unpack_archive(FILE_GZ_TO_DECOMPRESS, PLAYERS_DATA_PATH)
        #Get files uncompressed
        files = pathlib.Path(PLAYERS_DATA_PATH).glob('*.csv')
        for f in files:
            if f != pathlib.Path(PLAYER_BOXSCORE_PATH):
                os.remove(f)

    def initiate_dashboard(self):
        if self.boxscore is None:
            self.uncompress_file()
            #Players Data
            self.boxscore = pd.read_csv(PLAYER_BOXSCORE_PATH, sep=';', index_col=False)
            self.unique_players = self.boxscore['Player'].unique()
            os.remove(PLAYER_BOXSCORE_PATH)

            # return boxscore, unique_players

    def highlight(self, val):
        color = 'red' if val == 'L' else 'green'
        return f'background-color: {color}'

    def dashboard_first_row(self):
        #Row1----------------------------------------------
        row1_1, row1_spacer, row1_2 = st.beta_columns([1, 2, 1])

        with row1_1:
            self.player_selected = st.selectbox('Player', options=self.unique_players)
            self.number_of_game = int(self.boxscore['Player'].value_counts()[self.player_selected])

        with row1_2:
            self.number_selected = st.slider('Number of Games', max_value=self.number_of_game, value=20)
        # return player_selected, number_selected

    def dashboard_second_row(self):
        #Row2-----------------------------------------------
        # row2_1, row2_spacer, row2_2 = st.beta_columns([2, 0.5, 2])

        # Game Log
        player_data = self.boxscore.loc[self.boxscore['Player'] == self.player_selected].copy()

        player_data.drop(columns=['Unnamed: 0', 'Player', 'Player_Id', 'Team_Id', 'Game_Id'], inplace=True)
        player_data['Game Date'] = player_data['Game Date'].map(lambda x: datetime.strptime(x, DATE_FORMAT).date())
        player_data.sort_values(by='Game Date', ascending=False, inplace=True)
        player_data.reset_index(drop=True, inplace=True)
        player_data[['W/L']].style.applymap(self.highlight)
        st.dataframe(player_data.iloc[:self.number_selected])