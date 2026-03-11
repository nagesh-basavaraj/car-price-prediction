import kagglehub
import os
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor


# Download dataset
path = kagglehub.dataset_download("nehalbirla/vehicle-dataset-from-cardekho")

print("Dataset Path:", path)

# Load dataset
file_path = os.path.join(path, "car data.csv")
data = pd.read_csv(file_path)

print(data.head())


# Feature Engineering
current_year = 2024
data["CarAge"] = current_year - data["Year"]


# Encode categorical columns
fuel_encoder = LabelEncoder()
trans_encoder = LabelEncoder()

data["Fuel_Type"] = fuel_encoder.fit_transform(data["Fuel_Type"])
data["Transmission"] = trans_encoder.fit_transform(data["Transmission"])


# Features and Target
X = data[["CarAge","Fuel_Type","Transmission","Owner","Kms_Driven"]]
y = data["Selling_Price"]


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train Model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)


# Save model and encoders
pickle.dump(model, open("car_price_model.pkl","wb"))
pickle.dump(fuel_encoder, open("fuel_encoder.pkl","wb"))
pickle.dump(trans_encoder, open("trans_encoder.pkl","wb"))

print("Model trained and saved successfully!")