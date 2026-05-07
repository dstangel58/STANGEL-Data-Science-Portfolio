import pandas as pd # imported libraries
import streamlit as st
import numpy as np
from pathlib import Path # create portable file path 
from sklearn.preprocessing import StandardScaler #scales data to similar metrics
from sklearn.cluster import KMeans # model for clustering
from sklearn.decomposition import PCA # used for dimensionality reduction 
from sklearn.metrics import silhouette_score #calculates sihouette score

base_dir = Path(__file__).resolve().parent
data_folder = base_dir / 'data'

with st.sidebar:
    st.title('Data Management')

    uploaded_file = st.file_uploader("Upload your own CSV", type=['csv']) # file uploader element for app

    file_options = ['Dataset_1.csv', 'Dataset_2.csv', 'Dataset_3.csv', 'Dataset_4.csv'] # Each dataset has similar data, but different categories
    selected_filename = st.selectbox('Or choose a pre-loaded dataset:', file_options) #select box for datasets 1-4

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file).dropna()
        st.success(f"Loaded: {uploaded_file.name}")
    else:
        try:
            df = pd.read_csv(data_folder / selected_filename).dropna() # eliminates null values in selected file 
        except FileNotFoundError:
            st.error(f"Could not find {selected_filename} in /data. Check the filename.")
            st.stop()

    df = df.drop(columns=['country'], errors='ignore') # drops country column to ensure numerical-only data
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    selected_features = st.multiselect(
        "Step 1: Select Features for Model",
        options=numeric_cols,
        default=[c for c in ['child_mort', 'income', 'life_expec', 'inflation'] if c in numeric_cols] #pre-loads these four features in the model
    )

    if len(selected_features) < 2:
        st.error("Please select at least 2 features to continue.")
        st.stop() #requires a minimum of 2 features for PCA to work 

st.header('Country Wellbeing Categorization: Unsupervised Machine Learning Algorithm') 

tab1, tab2, tab3 = st.tabs(['Raw Data', 'Cluster Model', 'Visualizations']) #Raw Data for transparency, Cluster Model for assignment, customizable visualizations to show relationships between selected variables

with tab1: 
    st.title('Raw Data View')
    st.dataframe(df)

with tab2:
    X = df[selected_features].dropna().values

    n_samples, n_features = X.shape
    n_pca_components = min(2, n_samples, n_features)

    if n_samples < 2: # warnings for when minimums in the model aren't met 
        st.error("Not enough rows to cluster. Please load a larger dataset.")
        st.stop()

    if n_features < 2:
        st.error("Please select at least 2 features for PCA to work.")
        st.stop()

    scaler = StandardScaler() #installs standard scaler 
    X_std = scaler.fit_transform(X)

    pca = PCA(n_components=n_pca_components)
    X_pca = pca.fit_transform(X_std) #displays affect of components 

    max_k = min(10, n_samples) 

    if max_k < 2:
        st.error("Not enough data rows to cluster. Please load a larger dataset.")
        st.stop()

    sil_scores = [] #silhouette score calculation
    k_range = range(2, max_k + 1)

    for i in k_range: # 
        km = KMeans(n_clusters=i, init='k-means++', random_state=42)
        labels = km.fit_predict(X_pca)
        n_unique_labels = len(np.unique(labels))

        # Silhouette requires between 2 and n_samples - 1 unique labels
        if 2 <= n_unique_labels <= n_samples - 1:
            score = silhouette_score(X_pca, labels)
        else:
            score = -1  # assign worst possible score so it's never chosen as best_k

        sil_scores.append(score)

    # Filter to only valid scores when finding best_k
    valid_scores = [(k, s) for k, s in zip(k_range, sil_scores) if s > -1]

    if valid_scores:
        best_k = max(valid_scores, key=lambda x: x[1])[0]
    else:
        best_k = 2
        st.warning("Could not compute silhouette scores — defaulting to 2 clusters.")

    st.subheader('Silhouette Score')
    st.caption(f"Higher is better. Suggested k: **{best_k}** (highest score: {max(sil_scores):.3f})")
    sil_df = pd.DataFrame({'Number of Clusters': list(k_range), 'Silhouette Score': sil_scores})
    st.line_chart(sil_df, x='Number of Clusters', y='Silhouette Score')

    if max_k == 2: #creates slider for number of clusters + guard 
        chosen_k = 2
        st.info("Only 2 clusters possible with this dataset.")
    else:
        chosen_k = st.slider("Select number of clusters:",
                             min_value=2,
                             max_value=max_k,
                             value=best_k,       
                             step=1,
                             key='main_key')

    kmeans = KMeans(n_clusters=chosen_k, random_state=42) #creates KMeans model based on slider for clusters 
    clusters = kmeans.fit_predict(X_pca) 

    st.header('PCA Component Composition') #shows how much each component affects cluster formation
    st.write(pd.DataFrame(pca.components_, columns=selected_features))

    if n_pca_components == 2: # charts clusters 
        st.subheader('2D PCA Projection')
        pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
        pca_df['Cluster'] = clusters.astype(str)
        st.scatter_chart(pca_df, x='PC1', y='PC2', color='Cluster')
    else:
        st.info("Only 1 PCA component available — 2D plot not possible with this selection.")

    st.header('Country Category Predictor')
    st.write("Adjust the features to see what cluster a country would fall into!")

    user_inputs = [] #creates flexible st.sliders 
    for col in selected_features:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())

        if min_val == max_val:
            st.warning(f"'{col}' has no variation (all values are {min_val}) — skipping slider.")
            user_inputs.append(min_val)
        else:
            val = st.slider(
                label=f'{col.capitalize()} Value',
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                key=f'key_{col}'
            )
            user_inputs.append(val)

    pred_data = np.array([user_inputs])
    scaled_pred_data = scaler.transform(pred_data)
    pca_pred_data = pca.transform(scaled_pred_data)
    cluster_id = kmeans.predict(pca_pred_data)[0]

    st.divider()
    st.info(f"This input was assigned to **Cluster {cluster_id}**.")

    if n_pca_components == 2:
        st.subheader('Centroids Mapped in 2D Space') # maps centroids as coordinates 
        pc_centroid_df = pd.DataFrame(kmeans.cluster_centers_, columns=['PC1', 'PC2'])
        pc_centroid_df['Cluster'] = [f'Cluster {i}' for i in range(chosen_k)]
        st.scatter_chart(pc_centroid_df, x='PC1', y='PC2', color='Cluster')

        st.write("**Cluster centroids in PCA space:**")
        st.dataframe(pc_centroid_df.round(3))

with tab3:
    if len(selected_features) < 2:
        st.warning("Please select at least 2 features in the sidebar to view charts.") #Requires a minimum of two features to generate a scatter plot
    else:
        feature_1 = selected_features[0] #creates charts that update based on selected features 
        feature_2 = selected_features[1]

        st.title(f'{feature_1.capitalize()} vs. {feature_2.capitalize()}')
        st.scatter_chart(data=df, x=feature_1, y=feature_2)

        if len(selected_features) > 2: 
            feature_3 = selected_features[2]
            st.title(f'{feature_1.capitalize()} vs. {feature_3.capitalize()}')
            st.scatter_chart(data=df, x=feature_1, y=feature_3)