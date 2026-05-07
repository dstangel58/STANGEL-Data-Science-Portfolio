# 🏛️ **Congress 1984: Voting Records & Party Prediction** 🏛️

## **Project Summary:** 
This app uses Streamlit to display congressional voting data from 1984. You can even vote on bills like a member of Congress from that session! Based on 16 possible votes (including healthcare, trade, and foreign policy), you can view the decision tree classifier model to understand the level of partisanship in individual pieces of legislation.

Using multiple decision tree branches, this model shows which bills are the most polarized between the two parties. Some, like the physician fee freeze, have an obviously partisan split, while others are decidedly bipartisan. The model is optimized for its F1 score, prioritizing recall and precision equally.

## **Features:** 
* Exploratory Data Analysis: The raw dataset is converted from strings to floats using dictionaries within a dataframe.
* Machine Learning Pipeline: Optimized with hyperparameter tuning.
* Performance Metrics: Contains both a classification report and a confusion matrix.
* Interactive Quiz: Make 16 votes and let the model guess what your political affiliation would've been in 1984.

## 📖 **Steps to Use:** 📖
* Clone the Repository
    * git clone https://github.com/dstangel58/STANGEL-Data-Science-Portfolio.git
    * cd STANGEL-Data-Science-Portfolio/sml_streamlit_app
* Download requirements.txt
    * (Requires: pandas, streamlit, scikit-learn, matplotlib, pathlib, tomli, and time)
    * pip install -r requirements.txt
* Run the code!
    * streamlit run app.py

## 🧑‍🏫 **Guides and Tutorials:** 🧑‍🏫
* Data to Viz (Visualizations): https://www.data-to-viz.com/
* Streamlit Guide (Functions & Radio Buttons): https://streamlit.io/

## 📸 **Images:** 📸
![alt text](<Screenshot 2026-05-06 at 11.13.38 PM.png>)
Decision Tree Classifier Visual
![alt text](<Screenshot 2026-05-06 at 11.14.00 PM.png>)
Performance Metrics (Confusion Matrix)