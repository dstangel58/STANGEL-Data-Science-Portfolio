import pandas as pd 
import streamlit as st 
import seaborn as sns
import numpy as np
import sklearn as sk
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pathlib
import tomli as tomllib  
import time as time

st.header('Congress in 1984')
st.markdown('At the tail-end of the Cold War, the US Congress was grappling with a divided government, environmental damange, foreign trade, and intervention in Central America. Through this model, you get the opportunity to cast your own ballot!')


# 1. Get the directory where THIS script (app_converted.py) is located
# __file__ is a built-in variable that points to the current file
base_dir = pathlib.Path(__file__).parent 
csv_path = base_dir / 'congressional_voting_records.csv'

# 2. Build the path to the config file
# We use the / operator to join the current folder with the subfolders
config_path = base_dir/'.streamlit'/'config.toml'

# 3. Load the file safely
try:
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
except FileNotFoundError:
    st.error(f"Could not find config.toml at {config_path}. Check your folder structure!")
    config = {} # Create an empty dict so the rest of the app doesn't crash

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Raw Data', 'Model', 'Classification Report', 'Confusion Matrix', 'Quiz'])
with tab1: 
    st.header('Raw Data')
    st.write('View the original dataset here.')
    @st.cache_data
    def load_and_clean_data():
        csv_path = base_dir / 'congressional_voting_records.csv'
        df = pd.read_csv(csv_path).dropna()
        return df
    df = load_and_clean_data()
    st.dataframe(df, use_container_width=True)
with tab2: 
    st.header('Model')
    st.write('Take a look at the optimized decision tree model below.')
    # Preprocessing
    features = df[['handicapped-infants', 'water-project-cost-sharing', 'physician-fee-freeze', 
               'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban',
               'aid-to-nicaraguan-contras', 'mx-missile', 'immigration', 'synfuels-corporation-cutback',
               'education-spending', 'superfund-right-to-sue', 'crime', 'duty-free-exports', 
               'export-administration-act-south-africa']] #laws can be more broadly generalized into buckets like "environmental_regulation" and "trade_deals", this allows for more generalizability for future datasets
    X = features
    Y = df['party']
    # Model Creation 
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score

    vote_map = {
        'democrat' : 0.0,
        'republican' : 1.0
        } # converts strings to floats 
    feature_map = {
        'y' : 1.0,
        'n' : 0.0,
        '?' : 0.5
        } # converts strings to floats 

    X_numeric = X.replace(feature_map) # incorporates dictionaries 
    Y_numeric = Y.replace(vote_map)

    X_train, X_test, Y_train, Y_test = train_test_split(X_numeric, Y_numeric,
                                                        test_size=0.2,
                                                        random_state=42)

    model = DecisionTreeClassifier(random_state = 42, max_depth = 5) #limits max depth to prevent overfitting 
    model.fit(X_train, Y_train)

    import graphviz 
    from sklearn import tree 

    dot_data = tree.export_graphviz(model, feature_names = X.columns.tolist(), class_names=['democrat', 'republican'], filled=True)
    graph = graphviz.Source(dot_data) 
    graph

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
    st.write("Best parameters:", grid_search.best_params_)
    st.write("Best cross-validation score:", grid_search.best_score_)

    best_model = grid_search.best_estimator_
    Y_pred = best_model.predict(X_test)
with tab3: 
    st.header("Classification Report:")
    report = classification_report(Y_test, Y_pred, output_dict=True)
    class_df = pd.DataFrame(report).transpose()
    st.table(class_df)
with tab4: 
    st.header("Confusion Matrix")
    st.markdown("This matrix shows the accuracy of the best possible model based on optimal hyperparameters")
    cm = confusion_matrix(Y_test, Y_pred)
    plt.figure(figsize=(8, 6))
    heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    st.pyplot(heatmap.figure)
with tab5: 
    st.header('Quiz Yourself')
    st.write('Take the Quiz!')
    st.header("Input Features") ## allows for radio button voting
    st.markdown('Here, you can vote on issues as if you were a member of Congress in 1984. Based on party positions at the time, the model will predict what political party you would have belonged to')
    handicapped_infants = st.radio("How would you vote on a bill making the neglect of children with disabilities a child abuse crime?", ['y', 'n'], key='infant_poll')
    water_project_cost_sharing = st.radio("How would you vote on a bill that shifts waterway projct funding away from the federal government and toward local sources?", ['y', 'n'], key='water_poll')
    adoption_of_the_budget_resolution = st.radio("How would you vote on a bill that reduces the budget deficit by increasing defense spending and restructuring entitlement?", ['y', 'n'], key='budget_poll')
    physician_fee_freeze = st.radio("How would you vote on a bill that suspended medicare payments to physicians for 15 months to reduce inflation in expenditures?", ['y', 'n'],key='physicial_poll')
    el_salvador_aid = st.radio("How would you vote on a bill that supports the government of El Salvador during the civil war?", ['y', 'n'], key='el_salvador_poll')
    religious_groups_in_schools = st.radio("How would you vote on a bill that ensures religious student organizations have equal access to resources as nonreligious organizations?", ['y', 'n'], key='religious_groups_poll')
    anti_satellite_test_ban = st.radio("How would you vote on a bill that halts the testing of satellite weapons in space?", ['y', 'n'], key='satellite_poll')
    aid_to_nicaraguan_contras = st.radio("How would you vote on legislation that limits aid to Nicaraguan Contras?", ['y', 'n'], key='contras_poll')
    mx_missile = st.radio("How would you vote on the program to support MX (Peacekeeper) Missiles?", ['y', 'n'], key='mx_poll')
    immigration = st.radio("How would you vote on immigration legislation that provides sanctions for employers that knowingly hire undocumented immigrants, amnesty for some undocumented groups, a guest worker program for agriculture, and increased funding for broder protection services?", ['y', 'n'], key='immigration_poll')
    synfuels_corporation_cutback = st.radio("How would you vote on a bill that cutback on US projects exploring synthetic fuels?", ['y', 'n'], key='synfuels_poll')
    education_spending = st.radio("How would you vote on a bill that supports magnet programs alongside math, science, and foreign language instruction?", ['y', 'n'], key='education_poll')
    superfund_right_to_sue = st.radio("How would you vote on a bill that allows citizens to sue for violations of Superfund law with a longer time horizon after exposure, but lacking compensation provisions for victims of toxic waste exposure?", ['y', 'n'], key='superfund_poll')
    crime = st.radio("How would you vote on a bill that implements mandatory minimum sentences, abolished federal parole, and allows for limited pretrial detention?", ['y', 'n'], key='crime_poll')
    duty_free_exports = st.radio("How would you vote on aa bill that reduces duty-free access for developing nations in an effort to reduce trade deficits?", ['y', 'n'], key='exports_poll')
    export_administration_act_south_africa = st.radio("How would you vote on a bill that tightened rules regarding the export of dual-use (civilian that could be converted to military) technologies?", ['y', 'n'], key='export_poll')

    input_data = {
        'handicapped-infants': 1.0 if handicapped_infants == 'y' else 0.0,
        'water-project-cost-sharing': 1.0 if water_project_cost_sharing == 'y' else 0.0,
        'physician-fee-freeze': 1.0 if physician_fee_freeze == 'y' else 0.0,
        'el-salvador-aid': 1.0 if el_salvador_aid == 'y' else 0.0,
        'religious-groups-in-schools': 1.0 if religious_groups_in_schools == 'y' else 0.0,
        'anti-satellite-test-ban': 1.0 if anti_satellite_test_ban == 'y' else 0.0,
        'aid-to-nicaraguan-contras': 1.0 if aid_to_nicaraguan_contras == 'y' else 0.0,
        'mx-missile': 1.0 if mx_missile == 'y' else 0.0,
        'immigration': 1.0 if immigration == 'y' else 0.0,
        'synfuels-corporation-cutback': 1.0 if synfuels_corporation_cutback == 'y' else 0.0,
        'education-spending': 1.0 if education_spending == 'y' else 0.0,
        'superfund-right-to-sue': 1.0 if superfund_right_to_sue == 'y' else 0.0,
        'crime': 1.0 if crime == 'y' else 0.0,
        'duty-free-exports': 1.0 if duty_free_exports == 'y' else 0.0,
        'export-administration-act-south-africa': 1.0 if export_administration_act_south_africa == 'y' else 0.0
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df[X_test.columns]

    with st.spinner('Predicting your party...'):
        time.sleep(1) 
        st.success('Done!')

    probabilities = best_model.predict_proba(input_df) # uses input_df to data type errors 
    df_pred_prob = pd.DataFrame([input_data]) # Tied into radio button dictionary  

    display_df = pd.DataFrame({
        'Democrat': [probabilities[0][0]],
        'Republican': [probabilities[0][1]]
    }) # codes probability that an individual belongs to one party or another

    st.subheader('Predicted Political Party') # displays how much an individual is likely to belong to a party
    st.dataframe(
                display_df, 
                column_config={
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