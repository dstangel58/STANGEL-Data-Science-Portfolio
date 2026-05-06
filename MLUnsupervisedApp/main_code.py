import pandas as pd
import streamlit as st
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

base_dir = Path(__file__).resolve().parent
data_folder = base_dir / 'data'

with st.sidebar:
    st.title('Data Management')

    uploaded_file = st.file_uploader("Upload your own CSV", type=['csv'])

    file_options = ['Dataset_1.csv', 'Dataset_2.csv', 'Dataset_3.csv', 'Dataset_4.csv']
    selected_filename = st.selectbox('Or choose a pre-loaded dataset:', file_options)

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file).dropna()
        st.success(f"Loaded: {uploaded_file.name}")
    else:
        try:
            df = pd.read_csv(data_folder / selected_filename).dropna()
        except FileNotFoundError:
            st.error(f"Could not find {selected_filename} in /data. Check the filename.")
            st.stop()

    df = df.drop(columns=['gdpp', 'country'], errors='ignore')
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    selected_features = st.multiselect(
        "Step 1: Select Features for Model",
        options=numeric_cols,
        default=[c for c in ['child_mort', 'income', 'life_expec', 'inflation'] if c in numeric_cols]
    )

    if len(selected_features) < 2:
        st.error("Please select at least 2 features to continue.")
        st.stop()

st.header('Country Categorization: Unsupervised Machine Learning Algorithm')

tab1, tab2, tab3 = st.tabs(['Raw Data', 'Cluster Model', 'Visualizations'])

with tab1:
    st.title('Raw Data View')
    st.dataframe(df)

with tab2:
    X = df[selected_features].dropna().values

    n_samples, n_features = X.shape
    n_pca_components = min(2, n_samples, n_features)

    if n_samples < 2:
        st.error("Not enough rows to cluster. Please load a larger dataset.")
        st.stop()

    if n_features < 2:
        st.error("Please select at least 2 features for PCA to work.")
        st.stop()

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    pca = PCA(n_components=n_pca_components)
    X_pca = pca.fit_transform(X_std)

    max_k = min(10, n_samples)

    if max_k < 2:
        st.error("Not enough data rows to cluster. Please load a larger dataset.")
        st.stop()

    # --- Elbow plot using st.line_chart instead of matplotlib ---
    wcss = []
    for i in range(1, max_k + 1):
        km = KMeans(n_clusters=i, init='k-means++', random_state=42)
        km.fit(X_pca)
        wcss.append(km.inertia_)

    st.subheader('Elbow Score')
    elbow_df = pd.DataFrame({'Number of Clusters': range(1, max_k + 1), 'Inertia (WCSS)': wcss})
    st.line_chart(elbow_df, x='Number of Clusters', y='Inertia (WCSS)')

    if max_k == 2:
        chosen_k = 2
        st.info("Only 2 clusters possible with this dataset.")
    else:
        chosen_k = st.slider("Select number of clusters:",
                             min_value=2,
                             max_value=max_k,
                             value=min(5, max_k),
                             step=1,
                             key='main_key')

    kmeans = KMeans(n_clusters=chosen_k, random_state=42)
    clusters = kmeans.fit_predict(X_pca)

    st.header('PCA Component Composition')
    st.write(pd.DataFrame(pca.components_, columns=selected_features))

    # --- PCA scatter plot using st.scatter_chart instead of matplotlib ---
    if n_pca_components == 2:
        st.subheader('2D PCA Projection')
        pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
        pca_df['Cluster'] = clusters.astype(str)
        st.scatter_chart(pca_df, x='PC1', y='PC2', color='Cluster')
    else:
        st.info("Only 1 PCA component available — 2D plot not possible with this selection.")

    st.header('Country Category Predictor')
    st.write("Adjust the features to see what cluster a country would fall into!")

    user_inputs = []
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

    # --- Centroid plot using st.scatter_chart instead of matplotlib ---
    if n_pca_components == 2:
        st.subheader('Centroids Mapped in 2D Space')
        pc_centroid_df = pd.DataFrame(kmeans.cluster_centers_, columns=['PC1', 'PC2'])
        pc_centroid_df['Cluster'] = [f'Cluster {i}' for i in range(chosen_k)]
        st.scatter_chart(pc_centroid_df, x='PC1', y='PC2', color='Cluster')

        st.write("**Cluster centroids in PCA space:**")
        st.dataframe(pc_centroid_df.round(3))

    st.divider()
    st.info(f"This input was assigned to **Cluster {cluster_id}**.")

with tab3:
    if len(selected_features) < 2:
        st.warning("Please select at least 2 features in the sidebar to view charts.")
    else:
        feature_1 = selected_features[0]
        feature_2 = selected_features[1]

        st.title(f'{feature_1.capitalize()} vs. {feature_2.capitalize()}')
        st.scatter_chart(data=df, x=feature_1, y=feature_2)

        if len(selected_features) > 2:
            feature_3 = selected_features[2]
            st.title(f'{feature_1.capitalize()} vs. {feature_3.capitalize()}')
            st.scatter_chart(data=df, x=feature_1, y=feature_3)
    
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