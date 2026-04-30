import pandas as pd 
import streamlit as st
from pathlib import Path 

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent 
data_path = SCRIPT_DIR / 'combined_data' / 'merged_ESS5_reordered.csv'

@st.cache_data
def load_and_clean_data(path):
    df = pd.read_csv(path)
    
    # 1. Define the groups of columns
    vote_cols = [
        'prtvtcbe', 'prtvtbbg', 'prtvtcch', 'prtvtcy', 'prtvtbcz', 
        'prtvcde1', 'prtvcde2', 'prtvtbdk', 'prtvtcee', 'prtvtbes', 
        'prtvtbfi', 'prtvtbfr', 'prtvtgb', 'prtvtcgr', 'prtvthr', 
        'prtvtchu', 'prtvtaie', 'prtvtbil', 'prtvlt1', 'prtvlt2', 
        'prtvlt3', 'prtvtdnl', 'prtvtano', 'prtvtbpl', 'prtvtbpt', 
        'prtvtbru', 'prtvtase', 'prtvtcsi', 'prtvtbsk', 'prtvtbua'
    ]
    
    membership_cols = [
        'prtmbcbe', 'prtmbbbg', 'prtmbcch', 'prtmbcy', 'prtmbbcz', 'prtmbcde',
        'prtmbbdk', 'prtmbcee', 'prtmbbes', 'prtmbbfi', 'prtmbcfr', 'prtmbgb',
        'prtmbcgr', 'prtmbhr', 'prtmbchu', 'prtmbbie', 'prtmbbil', 'prtmblt', 
        'prtmbcnl', 'prtmbano', 'prtmbdpl', 'prtmbbpt', 'prtmbbru', 'prtmbase', 
        'prtmbcsi', 'prtmbbsk', 'prtmbcua'
    ]

    # 2. Collapse country columns into one single column per category
    # .bfill(axis=1) looks across the row and picks the first non-null value found
    df['Last Party Voted For'] = df[vote_cols].bfill(axis=1).iloc[:, 0]
    df['Party Membership'] = df[membership_cols].bfill(axis=1).iloc[:, 0]

    # 3. Drop the bulky original columns and weights
    cols_to_remove = vote_cols + membership_cols + ['dweight', 'pspwght', 'pweight', 'anweight']
    clean_df = df.drop(columns=cols_to_remove)
    
    return clean_df

# Execute cleaning
final_df = load_and_clean_data(data_path)

# --- Streamlit UI ---
st.title("ESS Data Cleaner")
st.write(f"Dataframe size: {final_df.shape[0]} rows and {final_df.shape[1]} columns.")

st.dataframe(final_df)

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv_data = convert_df(final_df)

st.download_button(
    label="📥 Download Cleaned 52k Row CSV",
    data=csv_data,
    file_name='ess_cleaned_optimized.csv',
    mime='text/csv',
)