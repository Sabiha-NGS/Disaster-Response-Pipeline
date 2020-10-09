# Disaster-Response-Pipeline

## About the Project

This project is about analyzing disaster data from Figure Eight to build 
a model for an API that classifies disaster messages.

## File Description

<b>Input file</b>

1.data/disaster_categories.csv  <i>Dataset with all the categories</i>
2.data/disaster_messages.csv	<i>Dataset with all the messages</i>

<b>Scripts</b>
data/process_data.py		<i>ETL script</i>
models/train_classifier.py	<i>Classification Pipeline</i>
app/run.py			<i>Flask file to run the app</i>	


### Instructions: (Run run.py directly if DisasterResponse.db and claasifier.pkl already exist.)
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/



