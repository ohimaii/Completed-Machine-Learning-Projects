# Disaster Response Pipeline Project
The purpose of this project is to use Natural Language Processing techniques to build a web application that aids people and/or organizations in responding to disaster reports. The web application takes in a user input message and classifies the message based on 36 Disaster Response categories

### Files and Folders
app

| - template : Contains HTML files for the web app
| |- master.html : Main page of web app
| |- go.html : Classification result page of web app
|- run.py : Flask file that runs app

data

|- disaster_categories.csv : Data to process
|- disaster_messages.csv : Data to process
|- process_data.py : Python script for data pre-processing and cleaning
|- InsertDatabaseName.db :  Database to save clean data to

models

|- train_classifier.py : Python script to train the ML Classifier
|- classifier.pkl : Saved model

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

DATA SOURCE: https://appen.com/datasets/combined-disaster-response-data/

