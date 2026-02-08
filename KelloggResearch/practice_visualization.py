import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

df = pd.read_csv(
    '/Users/duncanstangel/Documents/GitHub/STANGEL-Data-Science-Portfolio/KelloggResearch/ESS1e06_7-ESS2e03_6-ESS3e03_7-ESS4e04_6-ESS5e03_6-ESS6e02_7-ESS7e02_3-ESS8e02_3-ESS9e03_3-ESS10e03_3-ESS11e04_1-subset.csv'
)

# Filter Belgium + ESS round 1
be_df = df[(df['cntry'] == 'BE') & (df['essround'] == 1) & (df['prtvtbe'].notna())]

# Sample
smbe_df = be_df.sample(n=100, replace = False)

party_map = {
    1: "VLD",
    2: "CVP",
    3: "PS",
    4: "SP",
    5: "ECOLO",
    6: "VU-ID",
    7: "Agalev",
    8: "PRL-FDF",
    9: "Vlaams Blok",
    10: "PSC",
    66: "Other",
    77: "No answer",
    88: "Did not know",
    99: "No answer"
}

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
order = smbe_df['prtvtbe'].value_counts().index

sns.countplot(
    data=smbe_df,
    x='prtvtbe',
    order=order,
    ax=ax
)
labels = [party_map.get(i, i) for i in order]
ax.set_xticklabels(order, rotation=45, ha='right')
ax.set_xlabel("Party Voted For (prtvtbe)")
ax.set_ylabel("Number of Respondents")
ax.set_title("Belgium ESS Round 1 — Party Vote (Sample of 100)")

st.pyplot(fig)