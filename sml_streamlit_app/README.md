# 🏛️ Congress 1984: Voting Records & Party Prediction

This app uses streamlit to display congressional voting data from 1984. You can even vote on bills like a member of Congress from that session! Based on the 16 possible votes, you can view the decision tree classifier model to understand the level of partisanship in individual pieces of legislation.

# Features 

**Exploratory Data Analysis:** The raw dataset is converted from strings to floats using dictionaries, which is contained in a dataframe. There are 16 columns of data (including healthcare, trade, and foreign policy), each of which contains the entire Congress's vote on a certain piece of legislation. 
**Machine Learning Pipeline:** Optimized with hyperparameter tuning
**Performance Metrics:** Contains both a classification report and a confusion matrix 
**Interactive Quiz:** Make 16 votes and let the model guess what your political affiliation would've been in 1984. 


# The Model
![alt text](<Screenshot 2026-04-13 at 10.20.16 PM.png>)
Using multiple decision tree branches, this model shows which bills are the most polarized between the two parties. Some, like the physician fee freeze, have an obviously partisan split. Other bills are decidedly by partisan, with both Democrats and Republicans voting in favor (or against) a piece of legislation. The model is optimized for its F1 score, prioritizing recall and precision equally. 

# Running the Model 

1) Ensure the libraries in the requirements.txt file are downloaded locally using the pip install function (https://github.com/dstangel58/STANGEL-Data-Science-Portfolio/blob/main/sml_streamlit_app/requirements.txt). 
2) Run the code locally. 
3) Use cd to ensure presence in the proper folder. 
4) Use the streamlit runn app.py to activate the app in streamlit. 
5) Enjoy! 

# Other Links 

For visualizations: https://www.data-to-viz.com/
For streamlit functions (like radio buttons): https://streamlit.io/

This model requires the following libraries: pandas, streamlit, sk.learn, matplotlib.pyplot, pathlib, tomli, and time 