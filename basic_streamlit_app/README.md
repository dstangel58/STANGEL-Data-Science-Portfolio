## 🐧 Penguin Biology Dashboard 🐧

In this streamlit app, I use the penguins.csv dataset to compare various features including: bill length, bill depth, flipper length, among many more to discern trends. I display these using summary statistics, bar charts, and scatter plots. This shows the correlation between various features in the dataset.

# 🚀 Features 🚀 
**Filtering:** Uses the st.selectbox() function for all four categorical variables for maximum flexibility.
**Key Metrics:** Performs real-time calculations of averages and other metrics, updating for species and island respectively. 

# 📊 Visualization Options 📊
**Scatter Plots:** Exploring the correlation between bill depth and length (color-coded by sex).

**Distribution Analysis:** Seaborn boxplots comparing body mass across different islands.

**Trend Analysis:** Bar charts showing average bill length per Island.

# 🏗 Operating the Dataset 🏗
Dataset: penguins.csv 
Libraries: Pandas, Seaborn, Streamlit
Steps to Run: 
    1. Ensure you have the dataset: Place penguins.csv in a folder named data/.
    2. Install Libraries: pip install 
    3. pandas as pd 
    4. seaborn as sns
    5. streamlit as st
    6. ash, streamlit run main.py

# 📊 Visualization Key
_Quick summary of averages (Mean)_--**col#.metric** -- provides columns of summary statstics
_Scatter Plot_ -- **st.scatter_chart** -- Analyzing relationships between two continuous variables.
_Boxplot_ -- **sns.boxplot** -- Showing data distribution and outliers. (streamlit does not possess an in-built version) 
_Bar Chart_ -- **st.bar_chart** -- Comparing aggregated averages across categories.




