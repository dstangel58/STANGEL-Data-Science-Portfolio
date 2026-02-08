## 🐧 Penguin Biology Dashboard 🐧

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
Ensure you have the dataset: Place penguins.csv in a folder named data/.
Install Libraries: pip install 
pandas as pd 
seaborn as sns
streamlit as st
Bash, streamlit run main.py

# 📊 Visualization Key
_Quick summary of averages (Mean)_--**col#.metric** -- provides columns of summary statstics
_Scatter Plot_ -- **st.scatter_chart** -- Analyzing relationships between two continuous variables.
_Boxplot_ -- **sns.boxplot** -- Showing data distribution and outliers. (streamlit possesses not in-built version) 
_Bar Chart_ -- **st.bar_chart** -- Comparing aggregated averages across categories.




