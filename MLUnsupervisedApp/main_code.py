import pandas as pd 
import seaborn as sns 
import streamlit as st 
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from pathlib import Path 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA 

base_dir = Path(__file__).resolve().parent
data_path = base_dir / 'data' / 'Country-data.csv' 
import_data = pd.read_csv(data_path)

X = import_data[['child_mort','exports','health', 'income', 'imports','inflation','life_expec','total_fer','gdpp']].values 

scaler = StandardScaler()
X_std = scaler.fit_transform(X) 

k = 3 
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_std)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

plt.figure(figsize=(8,6))

for cluster_label in np.unique(clusters):
    indices = np.where(clusters == cluster_label)
    plt.scatter(X_pca[indices, 0], X_pca[indices, 1],
                alpha=0.7, edgecolor='k', s=60, label=f'Cluster {cluster_label}')
plt.xlabel('Principal Component 1') # life expectancy/income 
plt.ylabel('Principal Component 2') #primarily trade (import/export) focused 
plt.title('2D PCA Projection')
plt.legend(loc='best')
plt.grid(True)

# TABS 
    # Raw Data 
        # st.dataframe(import_data)
        # st.write('The first principal component explains variance through income and life expectancy, while the variance of the second principal component is largely defined by trade: imports and exports.')
        # st.write(pca.components_)
    # Income vs. Child Mortality (scatter plot in streamlit)
    # Life Expectancy vs. GDPP (scatterplot in streamlit) 
    # Cluster Model --> kmeans 
    # Slider Bars with Predictions --> https://docs.streamlit.io/develop/api-reference/widgets/st.slider 