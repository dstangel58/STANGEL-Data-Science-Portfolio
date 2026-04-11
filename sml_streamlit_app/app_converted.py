
# In[14]:


import pandas as pd 
import streamlit as st 
import numpy as np
import sklearn as sk


# In[ ]:


df = pd.read_csv('/Users/duncanstangel/Documents/GitHub/STANGEL-Data-Science-Portfolio/sml_streamlit_app/congressional_voting_records.csv').dropna()

features = df[['handicapped-infants', 'water-project-cost-sharing', 'physician-fee-freeze', 
               'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban',
               'aid-to-nicaraguan-contras', 'mx-missile', 'immigration', 'synfuels-corporation-cutback',
               'education-spending', 'superfund-right-to-sue', 'crime', 'duty-free-exports', 
               'export-administration-act-south-africa']] #laws can be more broadly generalized into buckets like "environmental_regulation" and "trade_deals", this allows for more generalizability for future datasets
X = features
Y = df['party'] 
df.head(5)


# In[102]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score


vote_map = {
    'democrat' : 0.0,
    'republican' : 1.0}
feature_map = {
    'y' : 1.0,
    'n' : 0.0,
    '?' : 0.5}

X_numeric = X.replace(feature_map)
Y_numeric = Y.replace(vote_map)

X_train, X_test, Y_train, Y_test = train_test_split(X_numeric, Y_numeric,
                                                    test_size=0.2,
                                                    random_state=42)

model = DecisionTreeClassifier(random_state = 42, max_depth = 4)
model.fit(X_train, Y_train)
model.feature_importances_


# In[ ]:


import graphviz 
from sklearn import tree 

dot_data = tree.export_graphviz(model, feature_names = X.columns.tolist(), class_names=['democrat', 'republican'], filled=True)
graph = graphviz.Source(dot_data)
graph


# In[ ]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [2, 3, 4, 5],
    'min_samples_split': [2, 3, 4, 5, 6],
    'min_samples_leaf': [2, 3, 4, 5, 6],
    'class_weight' : [None, 'balanced']
}

grid_search = GridSearchCV(estimator = model,
                           param_grid = param_grid,
                           cv = 5,
                           scoring = 'recall',
                           verbose = 3)
grid_search.fit(X_train, Y_train)
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)


# In[108]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

best_model = grid_search.best_estimator_

Y_pred = best_model.predict(X_test)

print("Classification Report:")
print(classification_report(Y_test, Y_pred))

cm = confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
st.pyplot(heatmap.figure)

st.header("Input Features") ## allows for radio button voting
handicapped_infants = st.radio("How would you vote on ____?", ['y', 'n'])
water_project_cost_sharing = st.radio("How would you vote on _____?", ['y', 'n'])
adoption_of_the_budget_resolution = st.radio("How would you vote on _____?", ['y', 'n'])
physician_fee_freeze = st.radio("How would you vote on _____?", ['y', 'n'])
el_salvador_aid = st.radio("How would you vote on _____?", ['y', 'n'])
religious_groups_in_schools = st.radio("How would you vote on _____?", ['y', 'n'])
anti_satellite_test_ban = st.radio("How would you vote on _____?", ['y', 'n'])
aid_to_nicaraguan_contras = st.radio("How would you vote on _____?", ['y', 'n'])
mx_missile = st.radio("How would you vote on _____?", ['y', 'n'])
immigration = st.radio("How would you vote on _____?", ['y', 'n'])
synfuels_corporation_cutback = st.radio("How would you vote on _____?", ['y', 'n'])
education_spending = st.radio("How would you vote on _____?", ['y', 'n'])
superfund_right_to_sue = st.radio("How would you vote on _____?", ['y', 'n'])
crime = st.radio("How would you vote on _____?", ['y', 'n'])
duty_free_exports = st.radio("How would you vote on _____?", ['y', 'n'])
export_administration_act_south_africa = st.radio("How would you vote on _____?", ['y', 'n'])

input_data = {
    'handicapped-infants': 1 if 'handicapped-infants' == 'y' else 0,
    'water-project-cost-sharing': 1 if 'water-project-cost-sharing' == 'y' else 0,
    'adoption-of-the-budget-resolution': 1 if 'adoption-of-the-budget-resolution' == 'y' else 0,
    'physician-fee-freeze': 1 if 'physician-fee-freeze' == 'y' else 0,
    'el-salvador-aid': 1 if 'el-salvador-aid' == 'y' else 0,
    'religious-groups-in-schools': 1 if 'religious-groups-in-schools' == 'y' else 0,
    'anti-satellite-test-ban': 1 if 'anti-satellite-test-ban' == 'y' else 0,
    'aid-to-nicaraguan-contras': 1 if 'aid-to-nicaraguan-contras' == 'y' else 0,
    'mx-missile': 1 if 'mx-missile' == 'y' else 0,
    'immigration': 1 if 'immigration' == 'y' else 0,
    'synfuels-corporation-cutback': 1 if 'synfuels-corporation-cutback' == 'y' else 0,
    'education-spending': 1 if 'education-spending' == 'y' else 0,
    'superfund-right-to-sue': 1 if 'superfund-right-to-sue' == 'y' else 0,
    'crime': 1 if crime == 'y' else 0,
    'duty-free-exports': 1 if 'duty-free_-xports' == 'y' else 0,
    'export-administration-act-south-africa': 1 if 'export-administration-act-south-africa' == 'y' else 0
}

input_df = pd.DataFrame([input_data])
input_df = input_df[X_test.columns]

Y_pred_prob = model.predict_proba(input_df) # uses user input to generate result; replace X_test with something relating to radio buttons

df_pred_prob = pd.DataFrame([input_data]) ## Basic structure; needs complexity from addition of radio buttons 

st.subheader('Predicted Political Party')
st.dataframe(column_config={
               'Democrat': st.column_config.ProgressColumn(
                 'Democrat',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               'Republican': st.column_config.ProgressColumn(
                 'Republican',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
             }, hide_index=True)