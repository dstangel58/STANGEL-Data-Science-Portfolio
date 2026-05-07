# 🐧 Penguin Biology Dashboard 🐧

## **Project Summary:**

This Streamlit application provides an interactive exploration of the `penguins.csv` dataset, allowing users to discern biological trends across species and habitats. By comparing features such as bill length, bill depth, and flipper length, the app uncovers correlations through real-time summary statistics and dynamic visualizations.

The dashboard leverages custom filtering for maximum flexibility, allowing users to slice data by species, island, or sex. Whether you are looking for specific flipper length trends or a high-level overview of body mass distribution, this tool provides a clean, automated interface for exploratory data analysis.

* **Filtering:** Uses `st.selectbox()` for all categorical variables, allowing for granular data isolation.
* **Real-time Metrics:** Dynamically calculates averages and key statistics that update instantly based on user selection.
* **Visual Synergy:** Combines built-in Streamlit charts with Seaborn’s advanced statistical plotting for a deeper look at outliers and distributions.

## 📖 **Steps to Use:** 📖

## **Clone the Repository**
* git clone [https://github.com/dstangel58/STANGEL-Data-Science-Portfolio.git](https://github.com/dstangel58/STANGEL-Data-Science-Portfolio.git)
    * cd STANGEL-Data-Science-Portfolio/basic_streamlit_app
* Download Libraries: 
    * Requires: pandas, seaborn, streamlit
* Run the code!
    * streamlit run main.py
## ⬇️ **Visualization Components:** ⬇️

* Scatter Plots (st.scatter_chart): Analyzes the correlation between bill depth and length, color-coded by sex.
* Distribution Analysis (sns.boxplot): Visualizes body mass across different islands to identify outliers.
* Trend Analysis (st.bar_chart): Displays aggregated averages for bill length per island.
* Key Metrics (st.metric): Provides a high-level summary of means and population counts.

## 🧑‍🏫 **Guides and Tutorials:** 🧑‍🏫

Streamlit Layouts (Columns & Metrics): https://docs.streamlit.io/develop/api-reference/layout/st.columns
Seaborn Boxplot Guide: https://seaborn.pydata.org/generated/seaborn.boxplot.html
Palmer Penguins Dataset Info: https://allisonhorst.github.io/palmerpenguins/

## 📸 **Images:** 📸

Charts