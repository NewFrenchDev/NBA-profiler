from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px 

from constants import *


FILES_DISPLAYED = ('Players boxscore', 'Players boxscore advanced',
                   'Players boxscore scoring', 'Players boxscore traditional')
FILES_PATH = {
    'Players boxscore': PLAYER_BOXSCORE_PATH,
    'Players boxscore advanced': PLAYER_BOXSCORE_ADVANCED_PATH,
    'Players boxscore scoring': PLAYER_BOXSCORE_SCORING_PATH,
    'Players boxscore traditional': PLAYER_BOXSCORE_TRADITIONAL_PATH
}

#Players Data
boxscore = pd.read_csv(PLAYER_BOXSCORE_PATH, sep=';', index_col=False)
boxscore_advanced = pd.read_csv(PLAYER_BOXSCORE_ADVANCED_PATH, sep=';', index_col=False)
boxscore_scoring = pd.read_csv(PLAYER_BOXSCORE_SCORING_PATH, sep=';', index_col=False)
boxscore_traditional = pd.read_csv(PLAYER_BOXSCORE_TRADITIONAL_PATH, sep=';', index_col=False)

@st.cache()
def load_csv(uploaded_file):
    csv = pd.read_csv(uploaded_file, sep=';')
    return csv

def display():

    dataframe_to_load = None

    import_or_use_local_dataset = st.sidebar.checkbox('Use NBA datasets')

    #Files user can used 
    if import_or_use_local_dataset:
        selected_file = st.sidebar.radio('Datasets', FILES_DISPLAYED)

        dataframe_to_load = pd.read_csv(FILES_PATH.get(selected_file), sep=';', index_col=False)
        dataframe_sampled = dataframe_to_load.sample(n=20000, random_state=200)
        dataframe_sampled['Game Date'] = dataframe_sampled['Game Date'].map(lambda x: datetime.strptime(x, DATE_FORMAT).date())
        dataframe_sampled.sort_values(by='Game Date', ascending=False, inplace=True)
        dataframe_sampled.drop(columns=['Unnamed: 0', 'Player_Id', 'Game_Id', 'Team_Id', 'Match Up', 'Game Date'], inplace=True, errors='ignore')
    else:
        #Upload dataset
        uploaded_file = st.sidebar.file_uploader('Upload your input CSV file', type=["csv"])

    # if selected_file == 'Players boxscore' and uploaded_file is None:
    #     dataframe_reduced = boxscore.iloc[:10000]
    #     st.dataframe(dataframe_reduced)
    # elif selected_file == 'Players boxscore' and uploaded_file is None:
    #     dataframe_reduced = boxscore_advanced.iloc[:10000]
    #     st.dataframe(dataframe_reduced)
    # if selected_file == 'Players boxscore' and uploaded_file is None:
    #     dataframe_reduced = boxscore.iloc[:10000]
    #     st.dataframe(dataframe_reduced)
    # if selected_file == 'Players boxscore' and uploaded_file is None:
    #     dataframe_reduced = boxscore.iloc[:10000]
    #     st.dataframe(dataframe_reduced)

        if uploaded_file is not None:
            dataframe_to_load = load_csv(uploaded_file)
        
            if len(dataframe_to_load) > 20000:
                dataframe_sampled = dataframe_to_load.sample(n=20000, random_state=200)
                dataframe_sampled.reset_index(drop=True, inplace=True)

    if dataframe_to_load is not None:
        st.header('**Input DataFrame**')
        st.write(dataframe_sampled)
        st.write('---')
        st.header('**Pandas Profiling Report**')
        st.write(' ')
        st.write('*Overview*')
        st.write(' ')
        st.header('***Dataset statistics***')
        number_of_columns = len(dataframe_to_load.columns)
        st.write(f'Number of column: {number_of_columns}')       
        number_of_observation = len(dataframe_to_load)
        st.write(f'Number of observations: {number_of_observation}')
        number_of_missing_values = sum(dataframe_to_load.isna().sum().values)
        st.write(f'Missing values: {number_of_missing_values}')        
        number_of_missing_values_percent = np.round(number_of_missing_values/(number_of_observation * number_of_columns) * 100, 2)
        st.write(f'Missing values(%): {number_of_missing_values_percent}%')
        number_of_duplicated_rows = len(dataframe_to_load) - len(dataframe_to_load.drop_duplicates())
        st.write(f'Duplicated rows: {number_of_duplicated_rows}')
        number_of_duplicated_rows_percent = np.round(number_of_duplicated_rows/(number_of_observation * number_of_columns) * 100, 2)
        st.write(f'Duplicated rows(%): {number_of_duplicated_rows_percent}%')
        total_size_memory_usage = np.round(dataframe_to_load.memory_usage(deep=True).sum() / 1000, 2)
        st.write(f'Total size memory usage: {total_size_memory_usage} KiB')
        average_size_memory_usage = np.round(dataframe_to_load.memory_usage(deep=True).mean() / 1000, 2)
        st.write(f'Average size in memory: {average_size_memory_usage} KiB')

        st.header('***Variable types***')
        types_in_dataframe = dataframe_to_load.dtypes.value_counts()
        for index, el in zip(types_in_dataframe.index, types_in_dataframe):
            st.write(f'{index}: {el}')

        st.header('***Variables***')    

        first_columns = dataframe_sampled.columns.values.tolist()[3:9]
        columns = st.multiselect(' ', options=dataframe_sampled.columns.values.tolist(), default=first_columns)

        for column in columns:
            col1_variable, col2_variable = st.beta_columns([1,3])
            # for column in dataframe_to_load.columns:

            with col1_variable:
                st.write(' ')
                st.header(column)
                serie = dataframe_to_load[column]
                distinct_values = len(serie.unique())
                st.write(f'Distinct values: {distinct_values}')
                distinct_values_percent = np.round((distinct_values / len(serie)) * 100, 2)
                st.write(f'Distinct values(%): {distinct_values_percent}')
                number_of_missing_values = serie.isna().sum()
                st.write(f'Missing values: {number_of_missing_values}')        
                number_of_missing_values_percent = np.round(number_of_missing_values/len(serie) * 100, 2)
                st.write(f'Missing values(%): {number_of_missing_values_percent}')
                if serie.dtype != object:
                    mean = np.round(serie.mean(), 2)
                    st.write(f'Mean: {mean}')
                    minimum = serie.min()
                    st.write(f'Minimum: {minimum}')
                    maximum = serie.max()
                    st.write(f'Maximum: {maximum}')
                memory_size = serie.memory_usage() / 1000
                st.write(f'Memory size: {memory_size} KiB')
            
            @st.cache(allow_output_mutation=True)
            def show_histogram(serie):
                fig = px.histogram(serie)
                return fig

            with col2_variable:
                fig = show_histogram(serie)
                st.plotly_chart(fig)

        st.write(' ')
        st.header('***Interactions***')

        col1_variable1, space1, col2_variable2, space2 = st.beta_columns([1, 0.2, 1, 2])
        
        with col1_variable1:
            selected_variable1 = st.selectbox('Variable1', options=dataframe_sampled.columns.values.tolist())
        with col2_variable2:
            selected_variable2 = st.selectbox('Variable2', options=dataframe_sampled.columns.values.tolist())

        fig_scatter = px.scatter(dataframe_sampled, x=selected_variable1, y=selected_variable2, hover_data=['Player'], color='Team')
        st.plotly_chart(fig_scatter)