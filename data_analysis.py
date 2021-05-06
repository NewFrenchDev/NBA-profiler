import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px 

from constants import *


FILES_DISPLAYED = ('Players boxscore', 'Players boxscore advanced',
                   'Players boxscore scoring', 'Player boxscore traditional')
FILES_PATH = {
    'Players boxscore': PLAYER_BOXSCORE_PATH,
    'Players boxscore advanced': PLAYER_BOXSCORE_ADVANCED_PATH,
    'Players boxscore scoring': PLAYER_BOXSCORE_SCORING_PATH,
    'Players boxscore traditional': PLAYER_BOXSCORE_TRADITIONAL_PATH
}

#Players Data
boxscore = pd.read_csv(PLAYER_BOXSCORE_PATH, sep=';', index_col=False)

@st.cache()
def load_csv(uploaded_file):
    csv = pd.read_csv(uploaded_file, sep=';')
    return csv

def display():

    #Files user can used 
    selected_file = st.sidebar.radio('Files', FILES_DISPLAYED)
    print(selected_file)

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
            dataframe_to_load = dataframe_to_load.sample(n=20000, random_state=200)

        st.header('**Input DataFrame**')
        st.write(dataframe_to_load)
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

        first_columns = dataframe_to_load.columns.values.tolist()[:4]
        columns = st.multiselect(' ', options=dataframe_to_load.columns.values.tolist(), default=first_columns)

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
            
            @st.cache
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
            selected_variable1 = st.selectbox('Variable1', options=dataframe_to_load.columns.values.tolist())
        with col2_variable2:
            selected_variable2 = st.selectbox('Variable2', options=dataframe_to_load.columns.values.tolist())

        fig_scatter = px.scatter(dataframe_to_load, x=selected_variable1, y=selected_variable2, hover_data=['Player'], color='Team')
        st.plotly_chart(fig_scatter)