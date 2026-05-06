import pandas as pd # install packages 
import seaborn as sns 
import streamlit as st 
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import toml
from pathlib import Path 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA 

base_dir = Path(__file__).resolve().parent
data_folder = base_dir / 'data'

# 2. Sidebar: Selection & Upload
with st.sidebar:
    st.title('Data Management')
        
    # File Uploader
    uploaded_file = st.file_uploader("Upload your own CSV", type=['csv'])
        
    # Pre-defined List
    file_options = [
        'Dataset_1',
        'Dataset_2', 
        'Dataset_3',
        'Dataset_4'
        ]
    selected_filename = st.selectbox('Or choose a pre-loaded dataset:', file_options)

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file).dropna()
    else:
        df = pd.read_csv(data_folder / selected_filename).dropna()

    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / 'data'

    # Drop non-numeric/unnecessary columns if they exist
    df = df.drop(columns=['gdpp', 'country'], errors='ignore')

    # 1. THE DYNAMIC SELECTION (Option #1)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_features = st.sidebar.multiselect(
        "Step 1: Select Features for Model",
        options=numeric_cols,
        default=[c for c in ['child_mort', 'income', 'life_expec', 'inflation'] if c in numeric_cols]
    )

    # Guard rail: Stop if nothing is selected
    if not selected_features:
        st.error("Please select at least one feature in the sidebar to continue.")
        st.stop()

st.header('Country Categorization: Unsupervised Machine Learning Algorithm')

tab1, tab2, tab3 = st.tabs(['Raw Data', 'Cluster Model', 'Visualizations'])

with tab1: 

    st.title('Raw Data View')
    st.dataframe(df)

with tab2: 

    X = df[selected_features].values
    
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    wcss = []
    k_range = range(1,11)

    for i in k_range: 
        kmeans_elbow = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans_elbow.fit(X_std)
        wcss.append(kmeans_elbow.inertia_)
    
    fig, ax = plt.subplots()
    ax.plot(k_range, wcss, marker='o')
    ax.set_title('Elbow Score')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Intertia (WCSS)')

    st.pyplot(fig, ax)

    chosen_k = st.slider("Select number of clusters based on the elbow above:", 
                         min_value=1,
                         max_value=10,
                         value=5,
                         step=1,
                         key='main_key')

    kmeans = KMeans(n_clusters=chosen_k, random_state=42)
    clusters = kmeans.fit_predict(X_std)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)

    st.header('PCA Component Composition')
    st.write(pca.components_)

    plt.figure(figsize=(8,6))
    fig, ax = plt.subplots()

    for cluster_label in np.unique(clusters):
        indices = np.where(clusters == cluster_label)
        plt.scatter(X_pca[indices, 0], X_pca[indices, 1],
                    alpha=0.7, edgecolor='k', s=60, label=f'Cluster {cluster_label}')
    plt.xlabel('Principal Component 1') # life expectancy/income 
    plt.ylabel('Principal Component 2') #primarily trade (import/export) focused 
    plt.title('2D PCA Projection')
    plt.legend(loc='best')
    plt.grid(True)
    st.pyplot(fig)

    st.header('Country Category Predictor')
    st.write("Adjust the features to see what cluster a country would fall into!")

    user_inputs = []

    for col in selected_features:
        min_val=float(df[col].min())
        max_val=float(df[col].max())
        mean_val=float(df[col].mean())

    for col in selected_features:
        val=st.slider(label=f'{col.capitalize()} Value',
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    key=f'key_{col}')
    user_inputs.append(val)

    pred_data = np.array([user_inputs])

    scaled_pred_data= scaler.transform(pred_data)
    prediction = kmeans.predict(scaled_pred_data)
    cluster_id=prediction[0]

    # ---  Cluster Labeling ---
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    pc_centroid_df = pd.DataFrame(centroids_pca, columns=['PC1', 'PC2'])
    
    pc_centroid_df['Cluster'] = [f'Cluster {i}' for i in range(chosen_k)]

    fig, ax = plt.subplots()

    ax.scatter(data=pc_centroid_df,
            x='PC1', 
            y='PC2',
            c='red',
            s=150,
            marker='X',
            edgecolor='k')
    
    st.pyplot(fig)
    
    for i, txt in enumerate(pc_centroid_df['Cluster']):
        plt.annotate(txt, (pc_centroid_df['PC1'][i], pc_centroid_df['PC2'][i]), 
                         xytext=(5, 5), textcoords='offset points')

    plt.title('Centroids Mapped in 2D space')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principle Component 2')

    # Show centroid table so you can calibrate thresholds
    st.write("**Cluster centroids in PCA space:**")
    st.dataframe(pc_centroid_df.round(3))

    # Display the final result with a clean UI
    st.divider()
    st.info(f"This input was assigned to **Cluster {cluster_id}**.")

with tab3: 
    feature_1=selected_features[0]
    feature_2=selected_features[1]

    st.title(f'{feature_1.capitalize()} vs. {feature_2.capitalize()}')
    st.scatter_chart(data=df, 
                     x=feature_1,
                     y=feature_2)

    if len(selected_features) > 2:
        feature_3 = selected_features[2]
        st.title(f'{feature_1.capitalize()} vs. {feature_3.capitalize()}')
        st.scatter_chart(data=df, 
                         x=feature_1,
                         y=feature_3)

        #(if pc1 > 1.5:
            #if pc2 > 0.5:
                #return "Developed / Trade-Open"
            #else:
                #return "Developed / Domestic-Focused"
        #elif pc1 < -1.0:
            #if pc2 > 0.5:
                #return "Fragile / Trade-Dependent"
            #else:
                #return "High-Need / Fragile"
        #else:
            #if pc2 > 1.0:
                #return "Emerging / Trade-Driven"
            #elif pc2 < -0.5:
                #return "Emerging / Import-Reliant"
            #else:
                #return "Middle-Income / Balanced")