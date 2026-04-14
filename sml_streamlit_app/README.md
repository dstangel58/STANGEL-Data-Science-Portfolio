# 🏛️ Congress 1984: Voting Records & Party Prediction

This app uses streamlit to display congressional voting data from 1984. You can even vote on bills like a member of Congress from that session! 

# Features 

**Exploratory Data Analysis:** The raw dataset is converted from strings to floats using dictionaries, which is contained in a dataframe. There are 16 columns of data (including healthcare, trade, and foreign policy), each of which contains the entire Congress's vote on a certain piece of legislation. 
**Machine Learning Pipeline:** Optimized with hyperparameter tuning
**Performance Metrics:** Contains both a classification report and a confusion matrix 
**Interactive Quiz:** Make 16 votes and let the model guess what your political affiliation would've been in 1984. 

# The Model
![alt text](<Screenshot 2026-04-13 at 10.20.16 PM.png>)
Using multiple decision tree branches, this model shows which bills are the most polarized between the two parties. Some, like the physician fee freeze, have an obviously partisan split. Other bills are decidedly by partisan, with both Democrats and Republicans voting in favor (or against) a piece of legislation. The model is optimized for its F1 score, prioritizing recall and precision equally. 

This model requires the following libraries: pandas, streamlit, sk.learn, matplotlib.pyplot, pathlib, tomli, and time 