############# DEPENDENCIES ###############
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.models import Variable
from datetime import datetime, timedelta
import requests
import json
import os

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from joblib import dump

from io import StringIO


############# DAG ###############

# DAG
my_dag = DAG(
    dag_id='metkapinezic',
    description='weather app exam',
    tags=['exam', 'datascientist'],
    schedule_interval='* * * * *',
    catchup=False,
    default_args={
        'owner': 'airflow',
        'start_date': datetime(2023, 9, 25),
    }
)

############# DEFINE TASKS ############### 

# TASK 1
def fetch_weather_data(**kwargs):
    API_KEY = "cf0dfbec12e5712f39b674cd89b7820e"
    CITIES = Variable.get("cities", default_var=['paris', 'london', 'washington'])  # Use a Variable to store cities

    output_directory = "/app/raw_files"  

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for city in CITIES:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()

            timestamp = datetime.now().strftime("%Y-%m-%d %H-%M")
            filename = os.path.join(output_directory, f"{city}_{timestamp}.json") 

            with open(filename, "w") as file:
                json.dump(data, file, indent=4)

            print(f"Data for {city} saved to {filename}")
        else:
            print(f"Failed to retrieve data for {city}: {response.status_code}")

# TASK 2
def read_raw_data_files(**kwargs):
    n_files = 20 
    parent_folder = "/app/raw_files"
    files = sorted(os.listdir(parent_folder), reverse=True)[:n_files]

    data_list = []  

    for f in files:
        with open(os.path.join(parent_folder, f), "r") as file:
            try:
                data_temp = json.load(file)  
            except json.JSONDecodeError:
                print(f"Error loading JSON from {f}: Invalid JSON or empty file")
                continue  

        # Check if data_temp is a list or a dictionary
        if isinstance(data_temp, list):
            for data_city in data_temp:
                data_list.append({
                    "temperature": data_city["main"]["temp"],
                    "city": data_city["name"],
                    "pressure": data_city["main"]["pressure"],
                    "date": f.split(".")[0],
                })
        elif isinstance(data_temp, dict):
            data_list.append({
                "temperature": data_temp["main"]["temp"],
                "city": data_temp["name"],
                "pressure": data_temp["main"]["pressure"],
                "date": f.split(".")[0],
            })
        else:
            print(f"Error processing data from {f}: Unexpected data format")

    return data_list  # Return data as a list of dictionaries

# TASK 3
def transform_data_into_csv(**kwargs):
    # Get the data passed from Task 2 using XCom
    ti = kwargs['ti']
    data_list = ti.xcom_pull(task_ids='read_raw_data')

    df = pd.DataFrame(data_list)

    output_folder = '/app/clean_data'

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filename = 'fulldata.csv'

    # Define the output path
    output_path = os.path.join(output_folder, filename)

    # Check if the CSV file already exists
    if os.path.exists(output_path):
        # If the file exists, read its current contents
        existing_df = pd.read_csv(output_path)

        # Append the new data to the existing DataFrame
        df = pd.concat([existing_df, df], ignore_index=True)

    # Extract the date from the filename correctly
    df['date'] = df['date'].str.split('_').str[-1]

    df.to_csv(output_path, index=False)

    # Push the output path to XCom for potential future use
    ti.xcom_push(key='output_path', value=output_path)

    print(f"Transformed data saved to {output_path}")

    # Add debugging log
    print(f"Output path pushed to XCom: {output_path}")

# TASK 4-1

def prepare_data_and_return_X_y(**kwargs):
    path_to_data = '/app/clean_data/fulldata.csv'
    df = pd.read_csv(path_to_data)
    
    df = df.sort_values(['city', 'date'], ascending=True)

    dfs = []

    for c in df['city'].unique():
        df_temp = df[df['city'] == c]

        # Creating target by shifting the temperature column
        df_temp['target'] = df_temp['temperature'].shift(1)

        # Creating features
        for i in range(1, 10):
            df_temp[f'temp_m-{i}'] = df_temp['temperature'].shift(-i)

        # Deleting rows with null values (due to lag)
        df_temp = df_temp.dropna()

        dfs.append(df_temp)

    # Concatenating datasets
    df_final = pd.concat(
        dfs,
        axis=0,
        ignore_index=False
    )

    # Deleting the 'date' variable
    df_final = df_final.drop(['date'], axis=1)

    # Creating dummies for the 'city' variable
    df_final = pd.get_dummies(df_final)

    # Separating the 'target' column
    target = df_final['target']

    # Dropping the 'target' column from features
    features = df_final.drop(['target'], axis=1)

    # Define output paths for features and target CSV files
    features_path = '/app/clean_data/features.csv'
    target_path = '/app/clean_data/target.csv'

    # Save features and target as CSV files
    features.to_csv(features_path, index=False)
    target.to_csv(target_path, index=False)

    # Push the file paths to XCom for later retrieval
    ti = kwargs['ti']
    ti.xcom_push(key='features_path', value=features_path)
    ti.xcom_push(key='target_path', value=target_path)

    # Debugging output
    print("Debug - prepare_data_and_return_X_y - X shape:", features.shape)
    print("Debug - prepare_data_and_return_X_y - y shape:", target.shape)

# TASK 4-2 Store X and y as XCom variables - TEST - NOT  IN USE
def store_X_y(ti):
    # Read X and y from XCom
    X_csv = ti.xcom_pull(key='X')
    y_csv = ti.xcom_pull(key='y')

    # Convert CSV strings back to DataFrames
    X = pd.read_csv(StringIO(X_csv))
    y = pd.read_csv(StringIO(y_csv))

    # Debugging output
    print("Debug - store_X_y - X shape:", X.shape)
    print("Debug - store_X_y - y shape:", y.shape)

    # You can use X and y as DataFrames in your task
    # For example: train your model using X and y

    return 'Data loaded from XCom'
   
# TASK 4-3

def train_and_save_models_without_store_X_y(**kwargs):
    ti = kwargs['ti']

    # Read features and target from the saved CSV files
    features_path = ti.xcom_pull(key='features_path', task_ids='prepare_data_and_return_X_y')
    target_path = ti.xcom_pull(key='target_path', task_ids='prepare_data_and_return_X_y')

    features = pd.read_csv(features_path)
    target = pd.read_csv(target_path)

    def train_and_save_model(model, X, y, path_to_model):
        # Training the model
        model.fit(X, y)
        # Saving the model
        print(str(model), 'saved at', path_to_model)
        dump(model, path_to_model)

    # Debugging output
    print("Debug - train_and_save_models - X shape:", features.shape)
    print("Debug - train_and_save_models - y shape:", target.shape)


    linear_regression_model = LinearRegression()
    train_and_save_model(linear_regression_model, features, target, '/app/linear_regression_model.pckl')

    decision_tree_model = DecisionTreeRegressor()
    train_and_save_model(decision_tree_model, features, target, '/app/decision_tree_model.pckl')

    random_forest_model = RandomForestRegressor()
    train_and_save_model(random_forest_model, features, target, '/app/random_forest_model.pckl')

# TASK 4-4: Compute model scores
def compute_model_scores(models, **kwargs):
    ti = kwargs['ti']
    X = ti.xcom_pull(key='X', task_ids='store_X_y')
    y = ti.xcom_pull(key='y', task_ids='store_X_y')

    if y is None:
        raise ValueError("Target variable 'y' is None. Make sure it's properly passed between tasks.")

    scores = {}
    
    for model in models:
        # computing cross val for each model
        cross_validation = cross_val_score(
            model,
            X,
            y,
            cv=3,
            scoring='neg_mean_squared_error')
        
        model_name = str(model)
        model_score = cross_validation.mean()
        scores[model_name] = model_score
    
    # Debugging output
    print("Debug - compute_model_scores - Model Scores:", scores)
    
    return scores

#TASK 5

def train_best_model(**kwargs):
    features_path, target_path = prepare_data_and_return_X_y(**kwargs)

    features = pd.read_csv(features_path)
    target = pd.read_csv(target_path)

    scores = compute_model_scores([LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor()], **kwargs)
    
    best_model = min(scores, key=scores.get)

    train_and_save_models_without_store_X_y(best_model, features, target, '/app/clean_data/best_model.pickle')
    
    print(f"Best model selected: {best_model}")


############# CREATE TASK ###############


# Create Task 1: Create a single task to fetch weather data for all cities
fetch_weather_task = PythonOperator(
    task_id='fetch_weather_data',
    python_callable=fetch_weather_data,
    provide_context=True,  # Pass the context to the function
    dag=my_dag,
)

# Create Task 2: Read Raw Data Files
task_read_raw_data = PythonOperator(
    task_id="read_raw_data",
    python_callable=read_raw_data_files,
    provide_context=True,
    dag=my_dag,
)

# Create Task 3: Transform Data into CSV
task_transform_data_into_csv = PythonOperator(
    task_id='transform_data_into_csv',
    python_callable=transform_data_into_csv,
    provide_context=True,
    dag=my_dag,
)


# Create Task 4-1: Prepare data
task_prepare_data_and_return_X_y = PythonOperator(
    task_id='prepare_data_and_return_X_y',
    python_callable=prepare_data_and_return_X_y,
    provide_context=True,
    dag=my_dag,
)


# Create Task 4-3: Train and save the model
task_train_and_save_models_without_store_X_y = PythonOperator(
    task_id='train_and_save_models_without_store_X_y',
    python_callable=train_and_save_models_without_store_X_y,
    provide_context=True,
    dag=my_dag,
)

# Create Task 4-4: Compute model score
models_to_evaluate = [LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor()]

task_compute_model_scores = PythonOperator(
    task_id='compute_model_scores',
    python_callable=compute_model_scores,
    op_args=[models_to_evaluate],
    provide_context=True,
    dag=my_dag,
)

#Create task 5
task_train_best_model = PythonOperator(
    task_id='train_best_model',
    python_callable=train_best_model,
    provide_context=True,
    dag=my_dag,
)

############# DAG DESIGN ###############

fetch_weather_task >> [task_read_raw_data, task_transform_data_into_csv]
task_transform_data_into_csv >> [task_prepare_data_and_return_X_y, task_train_and_save_models_without_store_X_y, task_compute_model_scores]
[task_prepare_data_and_return_X_y, task_train_and_save_models_without_store_X_y, task_compute_model_scores] >> task_train_best_model
