import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer as CT
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer as SI
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout
from scikeras.wrappers import KerasRegressor as kr
import joblib
# Importing Dataset
train_data = pd.read_csv('vehiclesFinal.csv')
train_data = train_data.drop(['id', 'manufacturer', 'model', 'region', 'lat', 'long', 'paint_color'], axis=1)
# Removing Cylinders text from column
train_data['cylinders'] = train_data['cylinders'].str.extract('(\d+)')
train_data['cylinders'] = train_data['cylinders'].ffill()
train_data['cylinders'] = train_data['cylinders'].astype(int)
train_data['year'] = train_data['year'].astype(int)
train_data['odometer'] = train_data['odometer'].astype(int)
# Extracting Numerical Features and Categorical Features
num_features = train_data.select_dtypes('number').columns
cat_features = train_data.select_dtypes(exclude='number').columns
# Handling Duplicates
train_data_duplicates = train_data.duplicated()
duplicate_rows = train_data[train_data.duplicated()]
all_duplicates = train_data[train_data.duplicated(keep=False)]
train_data_no_duplicates = train_data.drop_duplicates()
# Handling Outliers
for col in num_features:
    z_scores = np.abs((train_data[col] - train_data[col].mean()) / train_data[col].std())
    threshold = 3
    train_data = train_data[z_scores < threshold]
# Split features and target
X = train_data.drop('price', axis=1)
y = train_data['price']
# Define the Keras model
def create_model(input_dim):
    train_model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    train_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return train_model
# Extracting Features
numerical_features = X.select_dtypes('number').columns
categorical_features = X.select_dtypes(exclude='number').columns
# Creating Preprocessing Pipeline
preprocessing_pipeline = CT(transformers=[
    ('num', Pipeline([
        ('imputer', SI(strategy='mean')),
        ('scaler', StandardScaler())
    ]), numerical_features),
    ('cat', Pipeline([
        ('cat_imputer', SI(strategy='most_frequent')),
        ('encoding', OneHotEncoder())
    ]), categorical_features)
])
# train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Pipeline
pipeline = Pipeline(steps=[
    ('preprocessing_pipeline', preprocessing_pipeline),
    ('model', kr(model=create_model(43), epochs=20, batch_size=32, verbose=0))
])
# Train pipeline
pipeline.fit(X_train, y_train)
# Save the trained pipeline
joblib.dump(pipeline, 'price_prediction_pipeline.pkl')
joblib.dump(pipeline, 'price_prediction_pipeline.h5')
print("Pipeline saved!")
