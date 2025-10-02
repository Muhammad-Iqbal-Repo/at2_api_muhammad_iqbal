from fastapi import FastAPI, Query, HTTPException
from joblib import load
from pathlib import Path
from datetime import date
from datetime import datetime, timedelta
import pandas as pd

app = FastAPI()
# add constant
GITHUB_URL = "https://github.com/Muhammad-Iqbal-Repo/at2_api_muhammad_iqbal.git"

# get parents
BASE_DIR = Path(__file__).resolve().parent.parent

RAIN_MODEL_PATH = BASE_DIR / 'models' / 'rain.pkl'
PRECIPITATION_MODEL_PATH = BASE_DIR / 'models' / 'precipitation.pkl'
RAIN_FEATURES_PATH = BASE_DIR / 'data' / 'class_features.csv'
PRECIP_FEATURES_PATH = BASE_DIR / 'data' / 'regres_features.csv'

RAIN_TARGET = 'rain_in_7_days'
PRECIP_TARGET = 'volume_in_3_days'

rain_model = load(RAIN_MODEL_PATH)
precip_model = load(PRECIPITATION_MODEL_PATH)

# read the features for rain prediction

df_rain = pd.read_csv(RAIN_FEATURES_PATH)

# check time col
if 'time'not in df_rain.columns:
    raise ValueError("The 'time' column is missing in the features dataset.")

# ensuring time column is in datetime format
df_rain['time'] = pd.to_datetime(df_rain['time']).dt.date
df_rain.set_index('time', inplace=True)

df_precip = pd.read_csv(PRECIP_FEATURES_PATH)
# check time col
if 'time'not in df_precip.columns:
    raise ValueError("The 'time' column is missing in the features dataset.")

# ensuring time column is in datetime format
df_precip['time'] = pd.to_datetime(df_precip['time']).dt.date
df_precip.set_index('time', inplace=True)

# RAIN FEATURES

RAIN_FEATURES = [a for a in df_rain.columns if a != 'rain_in_7_days']
PRECIP_FEATURES = [a for a in df_precip.columns if a != 'volume_in_3_days']


@app.get("/", tags=["Root"])
def read_root():
    return {
            "Project" : "OPEN METEO - MACHINE LEARNING AS A SERVICE PROJECT FOR RAIN AND PRECIPITATION PREDICTION",
            "Objectives":
                {
                    "Binary Classification": "To predict the probability of rainfall in the next seven days",
                    "Regression": "To predict the cumulative precipitation in the next three days"
                },
            "List of Endpoints": {
                "Home" : {"method": "GET", "endpoint": "/"},
                "Health Check" : {"method": "GET", "endpoint": "/health"},
                "Rain Prediction" : {"method": "GET", 
                                    "endpoint": "/predict/rain",
                                    "Expected Date Input": {"date": "YYYY-MM-DD"},
                                    "Output_Example": {
                                        "Input Date": "2023-10-01",
                                        "Prediction":{
                                            "Date": "2023-10-08",
                                            "will_rain" : "True"
                                        }
                                    }
                },
                "Precipitation Prediction" : {"method": "GET", 
                                    "endpoint": "/predict/precipitation/fall",
                                    "Expected Date Input": {"date": "YYYY-MM-DD"},
                                    "Output_Example": {
                                        "Input Date": "2023-10-01",
                                        "Prediction":{
                                            "Start Date": "2023-10-04",
                                            "End Date": "2023-10-06",
                                            "precipitation_fall" : 5.67
                                        }
                                    }
                },
            },
            "Repository": GITHUB_URL,
            "Check Model Lists and Details": {"Method": "GET", "endpoint": "/models",
                                              "Output_Example": {
                                                "rain_model": "RandomForestClassifier",
                                                "precipitation_model": "RandomForestRegressor"
                                            }
            },
            "Check Features Used in Each Model": {"Method": "GET", 
                                                  "endpoint for rain features": "/rain/features",
                                                  "endpoint for precipitation features": "/precipitation/features",
                                                  "Output_Example": "The features are the same"
            }
        }
    
@app.get('/health', status_code=200, tags=["Health Check"])
def healthcheck():
    return "Welcome to the Prototype of OPEN METEO - MACHINE LEARNING AS A SERVICE PROJECT FOR RAIN AND PRECIPITATION PREDICTION"

@app.get('/models', tags=["Models"])
def list_models():
    return {
        "rain_model": f"The model for rain prediction is {rain_model.__class__.__name__}",
        "precipitation_model": f"The model for precipitation prediction is {precip_model.__class__.__name__}"
    }
    
 
# check features for rain model

@app.get("/rain/features", tags=["Features"])
def get_rain_features():
    target = RAIN_TARGET
    
    # get features from model
    model_features = rain_model.feature_names_in_.tolist()
    
    if target in RAIN_FEATURES:
        return (f"The target column '{target}' should not be included in the features list.")
    
    # check whether model features match the predefined features
    if set(model_features) != set(RAIN_FEATURES):
        raise HTTPException(status_code=500, detail="Mismatch between model features and predefined features.")
    else:
        return (f"The features are the same")
    
# check features for precipitation model

@app.get("/precipitation/features", tags=["Features"])
def get_precipitation_features():
    
    target = PRECIP_TARGET
        
    # get features from model
    model_features = precip_model.feature_names_in_.tolist()
    
    if target in PRECIP_FEATURES:
        return (f"The target column '{target}' should not be included in the features list.")
    
    
    # check whether model features match the predefined features
    # if there is mismatch print the difference
    if set(model_features) != set(PRECIP_FEATURES):
        raise HTTPException(status_code=500, detail="Mismatch between model features and predefined features.")
    else:
        return (f"The features are the same")
    
    

# rain prediction endpoint
@app.get("/prediction/rain/", tags=["Rain Prediction"])
@app.get("/predict/rain", tags=["Rain Prediction"])
def predict_rain(input_date: date = Query(..., alias="date", description="YYYY-MM-DD")):
    
    # validate date input
    if input_date not in df_rain.index:
        raise HTTPException(status_code=400, detail="Date not found in the dataset. Please provide a valid date.")
    
    # get features for the given date
    row = df_rain.loc[input_date, RAIN_FEATURES]
    if getattr(row, "ndim", 1) > 1: 
        row = row.iloc[0]
    X_df = pd.DataFrame([row.values], columns=RAIN_FEATURES)
    
    # make prediction
    try:
        y = rain_model.predict(X_df)
        will_rain = bool(int(round(float(y[0]))))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    
    prediction_date = input_date + timedelta(days=7)
    
    return {
        "Input Date": str(input_date),
        "Prediction": {
            "Date": str(prediction_date),
            "will_rain": will_rain
        }
    }

# precipitation prediction endpoint
@app.get("/prediction/precipitation/", tags=["Precipitation Prediction"])
@app.get("/predict/precipitation/fall", tags=["Precipitation Prediction"])
def predict_precipitation(input_date: date = Query(..., alias="date", description="YYYY-MM-DD")):
    # validate date input
    if input_date not in df_precip.index:
        raise HTTPException(status_code=400, detail="Date not found in the dataset. Please provide a valid date.")
    
    # get features for the given date
    row = df_precip.loc[input_date, PRECIP_FEATURES]
    if getattr(row, "ndim", 1) > 1:
        row = row.iloc[0]
    X_df = pd.DataFrame([row.values], columns=PRECIP_FEATURES)
    
    # make prediction
    try:
        y = precip_model.predict(X_df)
        precipitation_fall = float(round(float(y[0]), 2))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    
    start_date = input_date + timedelta(days=1)
    end_date = input_date + timedelta(days=3)
    
    return {
        "Input Date": str(input_date),
        "Prediction": {
            "Start Date": str(start_date),
            "End Date": str(end_date),
            "precipitation_fall": precipitation_fall
        }
    }