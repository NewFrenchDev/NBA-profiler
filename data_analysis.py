import shutil
import pathlib
import gc
import plotly
import base64

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px 
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from constants import *

FILENAME = 'Teams Data'
FILES_PATH = {
    'Teams Data': TEAMS_BOXSCORE_PATH,
}

class DataAnalysisBoard:

    def __init__(self, name):
        self.name = name
        self.dataframe = None
        self.dataframe_sampled = None

    def uncompress_file(self):
        shutil.unpack_archive(FILE_GZ_TO_DECOMPRESS, TEAMS_DATA_PATH)
        #Get files uncompressed
        self.dataframe = pd.read_csv(TEAMS_BOXSCORE_PATH, index_col="Unnamed: 0", dtype=DATAFRAME_COLUMNS_TYPE)
        #Transform the type of column Game date
        self.dataframe_sampled = self.dataframe.sample(1000)


    # def load_csv(self, uploaded_file):
    #     csv = pd.read_csv(uploaded_file)
    #     return csv

    def introduction(self):
        st.write("""
        Data science is a new weapon in sports and I will show you why!
        """)
        
    def display_dataframe(self):
        st.markdown('Dataset about 1000 random game on 49000 over the past 20 years')
        st.dataframe(self.dataframe_sampled)

    def display_team_evolution(self):
        #Figure des répartitions de W/L par équipe
        dataset_GSW = self.dataframe[self.dataframe['Team'] == "GSW"].loc[:]
        dataset_LAL = self.dataframe[self.dataframe['Team'] == "LAL"].loc[:]
        dataset_UTA = self.dataframe[self.dataframe['Team'] == "UTA"].loc[:]
        dataset_CHI = self.dataframe[self.dataframe['Team'] == "CHI"].loc[:]
        dataset_TOR = self.dataframe[self.dataframe['Team'] == "TOR"].loc[:]

        dataset_five_teams = pd.concat([dataset_GSW, dataset_LAL, dataset_UTA, dataset_CHI, dataset_TOR], ignore_index=True)
        dataset_five_teams.sort_values(by='Year', inplace=True)
        fig_evolution = px.histogram(dataset_five_teams, x='Team', y='W/L', color="W/L", barmode='group', animation_frame='Year')
        st.plotly_chart(fig_evolution)

        del fig_evolution
        gc.collect()

    def display_points_distribution(self):
        # data = self.dataframe.groupby(by='Year')['Points'].mean()
        fig_pts_dist = px.scatter(self.dataframe_sampled, x='Year', y='Points', color='W/L', marginal_x='box',  marginal_y='box')
        st.plotly_chart(fig_pts_dist)

        del fig_pts_dist
        gc.collect()


    def four_factor_view(self):
        st.header('The Four Factors')

        self.dataframe_sampled["Win"] = [0 if result == 'L' else 1 for result in self.dataframe_sampled["W/L"]]
        
        row1_1 , row1_space, row1_2 = st.beta_columns([2, 0.2, 2])

        #First factor
        with row1_1:
            fig1, axes = plt.subplots(1, 2, figsize=(15, 5))
            fig1.suptitle('FOUR FACTORS : Effective Field Goal Percentage')
            sns.regplot(ax=axes[0], x="Effective Field Goal Percentage", y="Win", data=self.dataframe_sampled, logistic=True)
            sns.regplot(ax=axes[1], x="Opponent Effective Field Goal Percentage", y="Win", data=self.dataframe_sampled, logistic=True)
            st.pyplot(fig1)

        #Second factor
        with row1_2:
            fig2, axes = plt.subplots(1, 2, figsize=(15, 5))
            fig2.suptitle('FOUR FACTORS : Turnover percentage')
            sns.regplot(ax=axes[0], x="Turnover percentage", y="Win", data=self.dataframe_sampled, logistic=True)
            sns.regplot(ax=axes[1], x="Opponent Turnover percentage", y="Win", data=self.dataframe_sampled, logistic=True)
            st.pyplot(fig2)

        row2_1 , row2_space, row2_2 = st.beta_columns([2, 0.2, 2])

        #Third factor
        with row2_1:
            fig3, axes = plt.subplots(1, 2, figsize=(15, 5))
            fig3.suptitle('FOUR FACTORS : Offensive rebounding percentage')
            sns.regplot(ax=axes[0], x="Offensive rebounding percentage", y="Win", data=self.dataframe_sampled, logistic=True)
            sns.regplot(ax=axes[1], x="Opponent Offensive rebounding percentage", y="Win", data=self.dataframe_sampled, logistic=True)
            st.pyplot(fig3)

        #Fouth factor
        with row2_2:
            fig4, axes = plt.subplots(1, 2, figsize=(15, 5))
            fig4.suptitle('FOUR FACTORS : Percent of Free Throws made')
            sns.regplot(ax=axes[0], x="Percent of Points (Free Throws)", y="Win", data=self.dataframe_sampled, logistic=True)
            sns.regplot(ax=axes[1], x="Opponent Percent of Points (Free Throws)", y="Win", data=self.dataframe_sampled, logistic=True)
            st.pyplot(fig4)

    def convert_image(self):
        with open(os.path.join(ROOT, 'images', 'newplot.png')) as f:
            data = f.read()
        return base64.b64encode(data).decode()

    def display_view(self):

        #Check if the dataframe is already in memory
        #If not load the data from file
        if self.dataframe is None:
            self.uncompress_file()

        self.introduction()
        st.write('---')
        self.display_dataframe()

        st.write('---')
        self.four_factor_view()

        self.display_points_distribution()

        self.display_team_evolution()

        path = os.path.join(ROOT, 'images', 'newplot.png')
        image = Image.open(path)
        st.image(image)

        #Always launch the garbage collector to free the memory
        gc.collect()