import pandas as pd
import numpy as np
import sklearn
import xgboost
import mlflow
import fastapi
import uvicorn

print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("sklearn:", sklearn.__version__)
print("xgboost:", xgboost.__version__)
print("mlflow:", mlflow.__version__)
print("fastapi:", fastapi.__version__)
print("uvicorn:", uvicorn.__version__)
print("All imports successful!")