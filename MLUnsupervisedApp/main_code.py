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

st.header('Country Categorization: Unsupervised Machine Learning Algorithm')

tab1, tab2, tab3 = st.tabs(['Raw Data', 'Cluster Model', 'Visualizations'])

with tab1: 
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / 'data' / 'Country-data.csv' 
    import_data = pd.read_csv(data_path).drop(columns='gdpp')

    X = import_data[['child_mort','exports','health', 'income', 'imports','inflation','life_expec','total_fer']].values 
    st.title('Raw Data')
    st.dataframe(import_data)

with tab2: 
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

    chosen_k = st.slider("Select number of clusters based on the elbow above:", 
                         min_value=1,
                         max_value=10,
                         value=5,
                         step=1)

    kmeans = KMeans(n_clusters=chosen_k, random_state=42)
    clusters = kmeans.fit_predict(X_std)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)

    st.write('The first principal component explains variance through income and life expectancy, while the variance of the second principal component is largely defined by trade: imports and exports.')
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


    # --- Child Mortality ---
    min_val = int(import_data['child_mort'].min())
    max_val = int(import_data['child_mort'].max())
    midpoint = (min_val + max_val) // 2

    child_mortality = st.slider(label='Child Mortality',
                                min_value=min_val,
                                max_value=max_val,
                                value=midpoint,
                                step=1,
                                key='key_child_mort')
    st.write(f'Child Mortality is equal to: {child_mortality} deaths per 1,000 live births')

    # --- Exports ---
    min_val = int(import_data['exports'].min())
    max_val = int(import_data['exports'].max())
    midpoint = (min_val + max_val) // 2

    exports = st.slider(label='Exports',
                        min_value=min_val,
                        max_value=max_val,
                        value=midpoint,
                        step=1,
                        key='key_exports')
    st.write(f'Exports are equal to: {exports} percent of GDP per capita')

    # --- Health ---
    min_val = int(import_data['health'].min())
    max_val = int(import_data['health'].max())
    midpoint = (min_val + max_val) // 2

    health = st.slider(label='Health',
                        min_value=min_val,
                        max_value=max_val,
                        value=midpoint,
                        step=1,
                        key='key_health')
    st.write(f'Health spending is equal to: {health} percent of GDP per capita')

    # --- Income ---
    min_val = int(import_data['income'].min())
    max_val = int(import_data['income'].max())
    midpoint = (min_val + max_val) // 2

    income = st.slider(label='Income',
                        min_value=min_val,
                        max_value=max_val,
                        value=midpoint,
                        step=1,
                        key='key_income')
    st.write(f'Income is equal to: {income} net dollars per person')

    # --- Imports ---
    min_val = int(import_data['imports'].min())
    max_val = int(import_data['imports'].max())
    midpoint = (min_val + max_val) // 2

    imports = st.slider(label='Imports',
                        min_value=min_val,
                        max_value=max_val,
                        value=midpoint,
                        step=1,
                        key='key_imports')
    st.write(f'Imports are equal to: {imports} percentage of GDP per capita')

    # --- Inflation ---
    min_val = int(import_data['inflation'].min())
    max_val = int(import_data['inflation'].max())
    midpoint = (min_val + max_val) // 2

    inflation = st.slider(label='Inflation',
                        min_value=min_val,
                        max_value=max_val,
                        value=midpoint,
                        step=1,
                        key='key_inflation')
    st.write(f'Inflation is equal to: {inflation} percentage of the growth rate in total GDP')

    # --- Life Expectancy ---
    min_val = int(import_data['life_expec'].min())
    max_val = int(import_data['life_expec'].max())
    midpoint = (min_val + max_val) // 2

    life_expectancy = st.slider(label='Life Expectancy',
                                min_value=min_val,
                                max_value=max_val,
                                value=midpoint,
                                step=1,
                                key='key_life_expec')
    st.write(f'Life Expectancy is equal to: {life_expectancy} years that one will live on average')

    # --- Total Fertility Rate ---
    min_val = int(import_data['total_fer'].min())
    max_val = int(import_data['total_fer'].max())
    midpoint = (min_val + max_val) // 2

    total_fer = st.slider(label='Total Fertility Rate',
                        min_value=min_val,
                        max_value=max_val,
                        value=midpoint,
                        step=1,
                        key='key_total_fer')
    st.write(f'Total Fertility Rate is equal to: {total_fer} children born per woman (if current age-fertility rates remain the same)')

    st.header('Cluster Prediction')

    pred_data = np.array(
        [[
            child_mortality,
            exports,
            health,
            income,
            imports,
            inflation,
            life_expectancy,
            total_fer
        ]]
    )
    
    scaled_pred_data= scaler.transform(pred_data)
    prediction = kmeans.predict(scaled_pred_data)
    cluster_id=prediction[0]

        # --- Dynamic Cluster Labeling ---
    centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
    centroid_df = pd.DataFrame(centroids_original, columns=[
        'child_mort', 'exports', 'health', 'income',
        'imports', 'inflation', 'life_expec', 'total_fer'
    ])

    def label_cluster(row):
        development_score = row['income'] + (row['life_expec'] * 500)
        trade_score = row['exports'] + row['imports']
        mortality_score = row['child_mort']

        if mortality_score > 80:
            return "High-Need / Fragile"
        elif development_score > 20000 and trade_score > 60:
            return "Developed / Trade-Open"
        elif development_score > 20000:
            return "Developed / Domestic-Focused"
        elif trade_score > 60:
            return "Emerging / Trade-Driven"
        elif row['inflation'] > 15:
            return "High-Inflation / Volatile"
        else:
            return "Developing / Average"
        
# Apply the labeling function to your centroid dataframe
    centroid_df['Label'] = centroid_df.apply(label_cluster, axis=1)

    # Fetch the label for the cluster the user's input belongs to
    predicted_label = centroid_df.iloc[cluster_id]['Label']

    # Display the final result with a clean UI
    st.divider()
    st.subheader("Final Categorization Result")
    st.success(f"### Category: **{predicted_label}**")
    st.info(f"This input was assigned to **Cluster {cluster_id}**.")

with tab3: 
    st.title('Child Mortality vs. Income')
    st.scatter_chart(data=import_data, 
                     x='income',
                     y='child_mort',
                     x_label='Income (net income per person in $)', 
                     y_label='Child Mortality (child death under 5 per 100,000 people)')

    st.title('Life Expectancy vs. Income')
    st.scatter_chart(data=import_data, 
                     x='income',
                     y='life_expec',
                     x_label='Income (net income per person in USD', 
                     y_label='Life Expectancy (Avg. mumber of years a child would live)')