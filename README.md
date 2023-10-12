# Airflow_WeatherApp

This project contains a DAG that retrieves information from an online weather data API, stores it, transforms it, and trains an algorithm on it.

The code is available in a python script dags/airflow_weather_app.py and contains 4 steps and 7 tasks.

#### Step 1: Data retrieval
The first task (1) retrieves the data from OpenWeatherMap, and stores a Variable named cities. This task also stores the data in JSON format in a file whose name corresponds to the time and date of the data collection: 2021-01-01 00:00.json. It creates a file in the /app/raw_files folder.
#### Step 2: Transformation
This step consists of two tasks that run simultaneously. Task (2) is reading the data and task (3) is transforming the data into csv format and outputs it in the /app/clean_data folder.
#### Step 3: Training prediction models
The tasks (4-1), (4-2), and (4-3) correspond to the training of different regression models (respectively LinearRegression, DecisionTreeRegressor, RandomForestRegressor). 
Task (4-1) reads the output /app/clean_data/fulldata.csv, cleans the data, and prepares it for the training of the ML models done in task (4-2).
Once these models have been trained they are tested with a cross-validation method in the task (4-3).
#### Step 4: Selecting the best model
The final step reads the output of the previous step in the format of scores, selects the model with minimum errors, trains it, and outputs the /app/clean_data
/best_model.pickle


