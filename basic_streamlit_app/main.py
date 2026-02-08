# import Libraries; each provides visualization importants (pandas and dataframes, seaborn boxplots, and streamlit visualization)
import pandas as pd 
import streamlit as st 
import seaborn as sns 

# import dataset; not specific to any one user; drops null values 
df = pd.read_csv("data/penguins.csv").dropna()

# App Heading
st.title("Penguin Biology By Island and Species")
st.markdown("This penguins dashboard provides an in-depth look into the penguins.csv dataset using only 120 lines of code. Through this exploratory data analysis, new trends can be discovered regarding factors such as flipper length and bill length/depth (across both species and island).")

# DataFrame Selection; four categorical selection choices; creates as much flexibility as possible--even possesses "All" option; 
island_options = ['All'] + list(df["island"].unique())
species_options = ['All'] + list(df['species'].unique())
sex_options = ['All'] + list(df['sex'].unique())
year_options = ['All'] + list(df['year'].unique().astype(str)) # Ensure years are strings for the list
island = st.selectbox("Select an Island", island_options)
species = st.selectbox("Select a Species", species_options)
sex = st.selectbox("Select a Sex", sex_options)
year = st.selectbox('Select a Year', year_options)
filter_df = df.copy() # uses if statement for "All" case
if island != "All":
    filter_df = filter_df[filter_df["island"] == island]
if species != "All":
    filter_df = filter_df[filter_df["species"] == species]
if sex != "All":
    filter_df = filter_df[filter_df["sex"] == sex]
if year != "All":
    filter_df = filter_df[filter_df["year"] == int(year)] # Cast back to int if needed

st.write(f"Showing results for: {species}, {island}, {sex}, {year}")
st.dataframe(filter_df)

# Setting up species-based categorization system and cleaning data; provides a dictionary that renames columns to better serve as graph axes
species_options = ['Adelie', 'Gentoo', 'Chinstrap']
selected_species = st.selectbox('Choose an option:', options=species_options)
filtered_df = df[df['species'] == selected_species]
filtered2_df = filtered_df.rename(columns={
    "ID": "Penguin Number",
    "species": "Species",
    "island": "Island",
    "bill_length_mm": "Bill Length (mm)",
    "bill_depth_mm": "Bill Depth (mm)",
    "flipper_length_mm": "Flipper Length (mm)",
    "body_mass_g": "Body Mass (g)",
    "sex": "Sex",
    "year": "Year"
})

# Summary Metrics (by Species); includes means for bill length and depth, average flipper length over a species; 
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Average Flipper Length Across Species", filtered2_df['Flipper Length (mm)'].mean().round(0))
col2.metric("Average Bill Length (mm)", round(filtered2_df["Bill Length (mm)"].mean()))
col3.metric("Average Bill Depth (mm)", round(filtered2_df["Bill Depth (mm)"].mean()))
col4.metric("Female %", round((filtered2_df["Sex"] == "female").mean()*100))
col5.metric("Male %", round((filtered2_df["Sex"] == "male").mean()*100))

# Depth vs Length; color determined by sex to show phenotypical differences between gender
st.title("Bill Depth vs. Bill Length")
st.scatter_chart(
    data = filtered2_df,
    x = "Bill Length (mm)",
    y = "Bill Depth (mm)",
    color = 'Sex'
)

# Shows average body mass by species to show size differences; uses seaborn for box plot functionality 
st.title("Average Body Mass by Species")
boxplot = sns.boxplot(
    x = "Island",
    y = "Body Mass (g)",
    data = filtered2_df
)
st.pyplot(boxplot.get_figure())

# Same as previous; quantifying phenotypical differences 
st.title("Average Bill Length by Island")
# Pre-calculate the mean to ensure it's not a total
avg_df = filtered2_df.groupby('Island', as_index=False)['Bill Length (mm)'].mean()
st.bar_chart(
    data=avg_df, 
    x='Island', 
    y='Bill Length (mm)'
)

st.write(f'You selected: {selected_species}')

# Island vs Body Mass; creates dictionary to rename variables to avoid renaming axis later on
# Uses separate filtration system for part 2 of app to avoid confusion 
island_options = ['Biscoe', 'Dream', 'Torgersen']
selected_island = st.selectbox('Choose an option:', options=island_options)
filtered3_df = df[df['island'] == selected_island]
filtered4_df = filtered3_df.rename(columns={
    "ID": "Penguin Number",
    "species": "Species",
    "island": "Island",
    "bill_length_mm": "Bill Length (mm)",
    "bill_depth_mm": "Bill Depth (mm)",
    "flipper_length_mm": "Flipper Length (mm)",
    "body_mass_g": "Body Mass (g)",
    "sex": "Sex",
    "year": "Year"
})

# Summary Metrics (by Island)
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Average Flipper Length Across Island", filtered4_df['Flipper Length (mm)'].mean().round(0))
col2.metric("Average Bill Length (mm)", round(filtered4_df["Bill Length (mm)"].mean()))
col3.metric("Average Bil Depth (mm)", round(filtered4_df["Bill Depth (mm)"].mean()))
col4.metric("Female %", round((filtered4_df["Sex"] == "female").mean()*100))
col5.metric("Male %", round((filtered4_df["Sex"] == "male").mean()*100))

# Analyzes Body Mass vs. Flipper Length; Island is individually selectable 
st.title("Flipper Length vs. Body Mass (g)")
st.scatter_chart(
    data = filtered4_df,
    x = "Flipper Length (mm)",
    y = "Body Mass (g)",
    color = 'Island'
)