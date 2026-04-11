
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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

best_model = grid_search.best_estimator_

Y_pred = best_model.predict(X_test)

print("Classification Report:")
print(classification_report(Y_test, Y_pred))


# Streamlit App Portion
# - Make a heatmap and horizontal bar chart instead

# In[ ]:

