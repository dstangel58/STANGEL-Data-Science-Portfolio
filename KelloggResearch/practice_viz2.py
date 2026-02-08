import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

df = pd.read_csv(
    '/Users/duncanstangel/Documents/GitHub/STANGEL-Data-Science-Portfolio/KelloggResearch/ESS1e06_7-ESS2e03_6-ESS3e03_7-ESS4e04_6-ESS5e03_6-ESS6e02_7-ESS7e02_3-ESS8e02_3-ESS9e03_3-ESS10e03_3-ESS11e04_1-subset.csv', usecols=['name', 'essround', 'idno', 'cntry', 'prtvtbe']
)

# Filter Belgium + ESS round 1
be_df = df[(df['cntry'] == 'BE') & (df['essround'] == 1)]

# Sample
smbe_df = be_df.sample(n=100, random_state=42)

st.bar_chart(
    smbe_df,
    x="prtvtbe",
    y="idno"
)
