# Generated from: ess5melted.ipynb
# Converted at: 2026-04-30T01:03:28.647Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import pandas as pd 
import seaborn as sns 
import streamlit as st
from pathlib import Path 


SCRIPT_DIR = Path(__file__).resolve().parent 
BASE_DIR = SCRIPT_DIR.parent 
data_path = SCRIPT_DIR / 'combined_data' / 'merged_ESS5_reordered.csv'

df = pd.read_csv(data_path)

cols_to_remove = ['dweight', 'pspwght', 'pweight', 'anweight']
filtered_df = df.drop(columns=cols_to_remove)

prtvt_df = filtered_df.melt(
    id_vars = ['idno', 'cntry'],
    value_vars = [
        'prtvtcbe', 'prtvtbbg', 'prtvtcch', 'prtvtcy', 'prtvtbcz', 
        'prtvcde1', 'prtvcde2', 'prtvtbdk', 'prtvtcee', 'prtvtbes', 
        'prtvtbfi', 'prtvtbfr', 'prtvtgb', 'prtvtcgr', 'prtvthr', 
        'prtvtchu', 'prtvtaie', 'prtvtbil', 'prtvlt1', 'prtvlt2', 
        'prtvlt3', 'prtvtdnl', 'prtvtano', 'prtvtbpl', 'prtvtbpt', 
        'prtvtbru', 'prtvtase', 'prtvtcsi', 'prtvtbsk', 'prtvtbua'
    ],
    var_name = 'Party Vote Code', # This tells you WHICH country column it came from
    value_name = 'Last Party Voted For'   # This is the ACTUAL value (e.g., 1.0, 5.0)
)

new_party_vote_values = {
    'prtvtcbe': 'Party Vote in Last Election in Belgium',
    'prtvtbbg': 'Party Vote in Last Election in Bulgaria',
    'prtvtcch': 'Party Vote in Last Election in Switzerland',
    'prtvtcy': 'Party Vote in Last Election in Cyprus', 
    'prtvtbcz': 'Party Vote in Last Election in Czechia',
    'prtvcde1': 'Party Vote in Last Election in Germany (Candidate)',
    'prtvcde2': 'Party Vote in Last Election in Germany (List)',
    'prtvtbdk': 'Party Vote in Last Election in Denmark',
    'prtvtcee': 'Party Vote in Last Election in Estonia',
    'prtvtbes': 'Party Vote in Last Election in Spain',
    'prtvtbfi': 'Party Vote in Last Election in Finland',
    'prtvtbfr': 'Party Vote in Last Election in France',
    'prtvtgb': 'Party Vote in Last Election in Great Britain', 
    'prtvtcgr': 'Party Vote in Last Election in Greece',
    'prtvthr': 'Party Vote in Last Election in Croatia',
    'prtvtchu': 'Party Vote in Last Election in Hungary',
    'prtvtaie': 'Party Vote in Last Election in Ireland',
    'prtvtbil': 'Party Vote in Last Election in Israel',
    'prtvlt1': 'Party Vote in Last Election in Lithuania (1)',
    'prtvlt2': 'Party Vote in Last Election in Lithuania (2)',
    'prtvlt3': 'Party Vote in Last Election in Lithuania (3)',
    'prtvtdnl': 'Party Vote in Last Election in Netherlands',
    'prtvtano': 'Party Vote in Last Election in Norway',
    'prtvtbpl': 'Party Vote in Last Election in Poland',
    'prtvtbpt': 'Party Vote in Last Election in Portugal',
    'prtvtbru': 'Party Vote in Last Election in the Russian Federation',
    'prtvtase': 'Party Vote in Last Election in Sweden',
    'prtvtcsi': 'Party Vote in Last Election in Slovenia',
    'prtvtbsk': 'Party Vote in Last Election in Slovakia', 
    'prtvtbua': 'Party Vote in Last Election in Ukraine'}

prtvt_df['Party Vote Code'] = prtvt_df['Party Vote Code'].replace(new_party_vote_values)

prtmb_df = df.melt(
    id_vars = ['idno', 'cntry'],
    value_vars = ['prtmbcbe', 'prtmbbbg', 'prtmbcch', 'prtmbcy', 'prtmbbcz', 'prtmbcde',
       'prtmbbdk', 'prtmbcee', 'prtmbbes', 'prtmbbfi', 'prtmbcfr', 'prtmbgb','prtmbcgr', 
       'prtmbhr', 'prtmbchu', 'prtmbbie', 'prtmbbil', 'prtmblt', 'prtmbcnl', 
       'prtmbano', 'prtmbdpl', 'prtmbbpt', 'prtmbbru', 'prtmbase', 'prtmbcsi', 'prtmbbsk',
       'prtmbcua'],
    var_name = 'Party Membership by Country', # This tells you WHICH country column it came from
    value_name = 'Party Membership by Voter')   # This is the ACTUAL value (e.g., 1.0, 5.0)

party_membership_dict = {'prtmbcbe': 'Party Membership in Belgium', 
                         'prtmbbbg': 'Party Membership in Bulgaria',
                         'prtmbcch': 'Party Membership in Switzerland', 
                         'prtmbcy': 'Party Membership in Cyrus', 
                         'prtmbbcz': 'Party Membership in Czechia', 
                         'prtmbcde': 'Party Membership in Germany',
                         'prtmbbdk': 'Party Membership in Denmark',
                         'prtmbcee': 'Party Membership in Estonia', 
                         'prtmbbes': 'Party Membership in Spain', 
                         'prtmbbfi': 'Party Membership in Finland', 
                         'prtmbcfr': 'Party Membership in France', 
                         'prtmbgb': 'Party Membership in Great Britain',
                         'prtmbcgr': 'Party Membership in Greece', 
                         'prtmbhr': 'Party Membership in Croatia', 
                         'prtmbchu': 'Party Membership in Hungary', 
                         'prtmbbie': 'Party Membership in Ireland', 
                         'prtmbbil': 'Party Membership in Israel', 
                         'prtmblt': 'Party Membership in Lithuania', 
                         'prtmbcnl': 'Party Membership in the Netherlands',
                         'prtmbano': 'Party Membership in Norway', 
                         'prtmbdpl': 'Party Membership in Poland', 
                         'prtmbbpt': 'Party Membership in Portugal', 
                         'prtmbbru': 'Party Membership in Russia', 
                         'prtmbase': 'Party Membership in Sweden', 
                         'prtmbcsi': 'Party Membership in Slovenia', 
                         'prtmbbsk': 'Party Membership in Slovakia',
                         'prtmbcua': 'Party Membership in Ukraine'}


# Instead of pd.concat, use a merge on the ID and Country columns
concat_df = pd.merge(
    prtvt_df, 
    prtmb_df, 
    on=['idno', 'cntry'], 
    how='inner'
)

# Now remove the redundant tracking columns you used for the melt
concat_df = concat_df.drop(['Party Membership by Country', 'Party Vote Code'], axis=1)

# Display result
st.dataframe(concat_df)
st.dataframe(concat_df)

# 1. Create a function to convert the dataframe to CSV
@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent it running on every rerun
    return df.to_csv(index=False).encode('utf-8')

# 2. Display the dataframe as usual
st.dataframe(concat_df)

# 3. Create the download button
csv_data = convert_df(concat_df)

st.download_button(
    label="📥 Download cleaned data as CSV",
    data=csv_data,
    file_name='european_voter_data.csv',
    mime='text/csv',
)