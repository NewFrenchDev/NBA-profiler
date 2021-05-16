import gc
import random

import streamlit as st
import pandas as pd
import plotly.express as px 
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns

from constants import *

class DataAnalysisBoard:

    def __init__(self, name):
        self.name = name
        self.dataframe = None
        self.dataframe_sampled = None

    def load_dataset(self, file_path):
        self.dataframe = pd.read_csv(file_path, index_col="Unnamed: 0", dtype=DATAFRAME_COLUMNS_TYPE)
        #Transform the type of column Game date
        self.dataframe_sampled = self.dataframe.sample(1000)
    
    def introduction(self):
        st.write("""
        Data science is a new weapon in sports and I will show you why with this WebApp! 😎 
        """)
        
    def display_dataframe(self):
        st.markdown(f'1000 random games from **{len(self.dataframe)}** over the past 20 years')
        st.dataframe(self.dataframe_sampled)

    def display_team_evolution(self, number_of_team):
        #Random team selected
        team_list = self.dataframe_sampled['Team'].unique().tolist()

        random_list_teams = []

        for _ in range(number_of_team):
            random_team = random.choice(team_list)
            team_list.remove(random_team)
            random_list_teams.append(random_team)

        #Teams selected dataset
        dataset_teams = self.dataframe.loc[self.dataframe['Team'].isin(random_list_teams)]

        #Processing
        dataset_teams["Win"] = [0 if result == 'L' else 1 for result in dataset_teams["W/L"]]
        dataset_teams.sort_values(by='Year', inplace=True)
        dataset_teams["Win average by year"] = dataset_teams.groupby(by=['Year', 'Team'])['Win'].transform('mean')
        dataset_teams["Effective Field Goal Percentage average by year"] = dataset_teams.groupby(by=['Year', 'Team'])['Effective Field Goal Percentage'].transform('mean')
        dataset_teams["Points average by year"] = dataset_teams.groupby(by=['Year', 'Team'])['Points'].transform('mean')
        dataset_teams["Points diff average by year"] = dataset_teams.groupby(by=['Year', 'Team'])['Plus-Minus Impact'].transform('mean')
        dataset_teams.drop_duplicates(subset=['Year', 'Team'])

        #Figures
        fig_win_average = px.line(dataset_teams, x='Year', y='Win average by year', color='Team',title=f'Win average over 20 years' ,)
        fig_efg_average = px.scatter(dataset_teams, x='Year', y='Effective Field Goal Percentage average by year', color='Team',
                                    title=f'Effective field goal average over 20 years' , marginal_y='box')
        fig_pts_average = px.scatter(dataset_teams, x='Year', y='Points average by year', color='Team',  marginal_y='box', title='Points average over 20 years')
        fig_pts_diff_average = px.box(dataset_teams, x='Year', y='Plus-Minus Impact', color='Team', boxmode='overlay', title='Points diff average overs 20 years' )

        #Display
        st.write('---')
        st.header("Teams's evolution over the years")

        row1_1, space, row1_2 = st.beta_columns([2,0.5,2])
        with row1_1:
            st.plotly_chart(fig_win_average, use_container_width=True)
        with row1_2:    
            st.plotly_chart(fig_efg_average, use_container_width=True)

        row2_1, space, row2_2 = st.beta_columns([2,0.5,2])
        with row2_1:      
            st.plotly_chart(fig_pts_average, use_container_width=True)
        with row2_2:
            st.plotly_chart(fig_pts_diff_average, use_container_width=True)

        del fig_win_average
        del fig_efg_average
        del fig_pts_average
        del fig_pts_diff_average
        gc.collect()

    def display_distribution(self, density_distribution_group):

        if len(density_distribution_group) > 10:
            st.warning("More than 10 at the same time won't help you to have a clear analysis!")
            st.stop()

        figs_col1 = []
        figs_col2 = []

        col1, space, col2 = st.beta_columns([2, 0.5, 2])

        #
        for i in range(0, len(density_distribution_group), 2):
            fig = ff.create_distplot([self.dataframe_sampled[density_distribution_group[i]]],
                                group_labels=[density_distribution_group[i]])
            figs_col1.append(fig)

        for i in range(1, len(density_distribution_group), 2):
            fig = ff.create_distplot([self.dataframe_sampled[density_distribution_group[i]]],
                                group_labels=[density_distribution_group[i]])
            figs_col2.append(fig)

        with col1:
            for fig in figs_col1:
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            for fig in figs_col2:
                st.plotly_chart(fig, use_container_width=True)

    def create_four_factor_regplot(self, title, team_column, opponent_column):

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle(title)
        sns.regplot(ax=axes[0], x=team_column, y="Win", data=self.dataframe_sampled, logistic=True)
        sns.regplot(ax=axes[1], x=opponent_column, y="Win", data=self.dataframe_sampled, logistic=True)

        return fig

    def generate_figures(self, titles, team_factors, opponent_factors):    
        figs = []
        number = 0

        #Indicate to user the progression
        my_progress_bar = st.progress(0)

        for title, team_factor, opponent_factor in zip(titles, team_factors, opponent_factors):
            fig = self.create_four_factor_regplot(title, team_factor, opponent_factor)
            figs.append(fig)
            #Update progress bar
            number += 1
            my_progress_bar.progress(number/4)

        return figs

    def display_four_factor(self):

        self.dataframe_sampled["Win"] = [0 if result == 'L' else 1 for result in self.dataframe_sampled["W/L"]]

        slot_success_message = st.empty()

        with st.spinner(text='Four Factors processing in progress...'):
            
            titles = ['Effective Field Goal percentage', 'Turnover percentage',
                      'Offensive rebounding percentage', 'Free Throw Rate']

            team_factors = ['Effective Field Goal Percentage', 'Turnover percentage',
                            'Offensive rebounding percentage', 'Free Throw Rate']

            opponent_factors = ['Opponent Effective Field Goal Percentage', 'Opponent Turnover percentage',
                                'Opponent Offensive rebounding percentage', 'Opponent Free Throw Rate']
      
            figs = self.generate_figures(titles, team_factors, opponent_factors)

            slot_success_message.success('Processing done!')

        st.write('---')
        st.header('The Four Factors')

        row1_1 , row1_space, row1_2 = st.beta_columns([2, 0.2, 2])

        #First factor
        with row1_1:       
            st.pyplot(figs[0])

        #Second factor
        with row1_2:  
            st.pyplot(figs[1])

        row2_1 , row2_space, row2_2 = st.beta_columns([2, 0.2, 2])

        #Third factor
        with row2_1:
            st.pyplot(figs[2])

        #Fouth factor
        with row2_2:
            st.pyplot(figs[3])
    

    def display_view(self, density_distribution_group, show_four_factors):

        #Check if the dataframe is already in memory
        #If not load the data from file
        if self.dataframe is None:
            self.load_dataset(file_path=TEAMS_BOXSCORE_PATH)

        self.introduction()

        self.display_dataframe()

        self.display_team_evolution(number_of_team=5)

        st.write("---")
        st.header("Density distribution")
        self.display_distribution(density_distribution_group)

        if show_four_factors:
            st.write('---')
            self.display_four_factor()

        #Always launch the garbage collector to free the memory
        gc.collect()