import streamlit as st 

# Markdown Hashtag 
st.title("Hello, Streamlit")
st.markdown("# Hello, Streamlit")

st.write("This is my first Streamlit app.")

import pandas as pd 

st.subheader("Exploring Our Dataset")

# Load in CSV File 

df = pd.read_csv("data/sample_data-1.csv")
st.write("Here's our Data!")
st.dataframe(df)

city = st.selectbox("Select a city", df["City"].unique())
filtered_df = df[df["City"] == city]

st.write(f"People in (City)") 
st.dataframe(filtered_df)

salary = st.selectbox("Select a salary", df["Salary"].unique())
st.bar_chart(df[df["Salary"] == salary])

import seaborn as sns 

box_plot1 = sns.boxplot(x = df["City"], y = df["Salary"])

st.pyplot(box_plot1.get_figure())

