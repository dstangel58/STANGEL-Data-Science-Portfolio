import pandas as pd 
import seaborn as sns 
import streamlit as st 
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans 

base_dir = Path('MLUnsupervisedApp').parent.resolve()
data_path = base_dir / 'data' / 'Country-data.csv' 
import_data = pd.read_csv(data_path)

X = import_data[['child_mort','exports','health','imports','income','inflation','life_expec','total_fer','gdpp']]

scaler = StandardScaler()
X_std = scaler.fit_transform(X) # add x variable 

k = 4 
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_std)

# TABS 
    # Raw Data 
    # Income vs. Child Mortality (scatter plot in streamlit)
    # Life Expectancy vs. GDPP (scatterplot in streamlit) 
    # Cluster Model --> kmeans 
    # Slider Bars with Predictions --> https://docs.streamlit.io/develop/api-reference/widgets/st.slider 