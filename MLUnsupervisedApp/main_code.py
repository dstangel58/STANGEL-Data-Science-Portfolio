import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
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
    X = df[selected_features].dropna().values  # --- added .dropna() here too to be safe

    # --- Guard: need at least 2 rows AND 2 features for PCA ---
    n_samples, n_features = X.shape
    n_pca_components = min(2, n_samples, n_features)  # --- never request more components than data allows

    if n_samples < 2:
        st.error("Not enough rows to cluster after removing missing values. Please load a larger dataset.")
        st.stop()

    if n_features < 2:
        st.error("Please select at least 2 features for PCA to work.")
        st.stop()

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # --- PCA moved before KMeans so n_pca_components is available for elbow loop ---
    pca = PCA(n_components=n_pca_components)  # --- dynamic instead of hardcoded 2
    X_pca = pca.fit_transform(X_std)

    max_k = min(10, n_samples)

    if max_k < 2:
        st.error("Not enough data rows to cluster. Please load a larger dataset.")
        st.stop()

    # Elbow loop runs on X_pca now that PCA is fitted
    wcss = []
    for i in range(1, max_k + 1):
        km = KMeans(n_clusters=i, init='k-means++', random_state=42)
        km.fit(X_pca)
        wcss.append(km.inertia_)

    fig, ax = plt.subplots()
    ax.plot(range(1, max_k + 1), wcss, marker='o')
    ax.set_title('Elbow Score')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Inertia (WCSS)')
    st.pyplot(fig)

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
    clusters = kmeans.fit_predict(X_pca)  # --- clustering on X_pca, consistent with elbow loop

    st.header('PCA Component Composition')
    st.write(pca.components_)

    # PCA scatter plot
    # --- Guard: only plot 2D if we actually have 2 components ---
    if n_pca_components == 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        for cluster_label in np.unique(clusters):
            indices = clusters == cluster_label
            ax.scatter(X_pca[indices, 0], X_pca[indices, 1],
                       alpha=0.7, edgecolor='k', s=60, label=f'Cluster {cluster_label}')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_title('2D PCA Projection')
        ax.legend(loc='best')
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.info("Only 1 PCA component available — 2D plot not possible with this selection.")

    st.header('Country Category Predictor')
    st.write("Adjust the features to see what cluster a country would fall into!")

    user_inputs = []
    for col in selected_features:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())

        # --- Guard: if min == max the slider will crash ---
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
    pca_pred_data = pca.transform(scaled_pred_data)      # --- project into PCA space before predicting
    prediction = kmeans.predict(pca_pred_data)
    cluster_id = prediction[0]

    # Centroid plot
    if n_pca_components == 2:
        pc_centroid_df = pd.DataFrame(kmeans.cluster_centers_, columns=['PC1', 'PC2'])
        pc_centroid_df['Cluster'] = [f'Cluster {i}' for i in range(chosen_k)]

        fig, ax = plt.subplots()
        ax.scatter(pc_centroid_df['PC1'], pc_centroid_df['PC2'],
                   c='red', s=150, marker='X', edgecolor='k')

        for i, txt in enumerate(pc_centroid_df['Cluster']):
            ax.annotate(txt, (pc_centroid_df['PC1'][i], pc_centroid_df['PC2'][i]),
                        xytext=(5, 5), textcoords='offset points')

        ax.set_title('Centroids Mapped in 2D Space')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        st.pyplot(fig)

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