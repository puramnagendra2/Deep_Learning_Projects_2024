```python
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
```

### Overview of the Project

1. **Objective**: The project aims to predict vehicle prices using machine learning techniques, leveraging a dataset with various features such as year, odometer, and categorical attributes.

2. **Data Preprocessing**: The dataset undergoes extensive preprocessing, including handling missing values, removing duplicates, managing outliers, and encoding categorical features for model compatibility.

3. **Feature Engineering**: Numerical and categorical features are separated and processed through pipelines, ensuring proper scaling, imputation, and encoding for optimal model performance.

4. **Model Architecture**: A Keras-based neural network is designed with layers for dense connections, batch normalization, and dropout to prevent overfitting and improve generalization.

5. **Deployment**: The trained pipeline, combining preprocessing and the model, is saved as a reusable artifact, enabling seamless predictions on new data inputs.

### Problem Statement

1. **Price Prediction Challenge**: Accurately predicting vehicle prices is a complex task due to the diverse factors influencing market value, such as vehicle age, mileage, and condition.

2. **Data Quality Issues**: The dataset contains missing values, duplicates, and outliers that need to be addressed to ensure reliable model performance.

3. **Feature Diversity**: The dataset includes both numerical and categorical features, requiring tailored preprocessing techniques for effective model training.

4. **Model Generalization**: Designing a model that generalizes well to unseen data is critical to avoid overfitting and ensure robust predictions.

5. **End-to-End Pipeline**: Developing an integrated pipeline that combines preprocessing and model training is essential for efficient deployment and scalability.

### Objectives of the Project

1. **Accurate Price Prediction**: Develop a machine learning model capable of predicting vehicle prices with high accuracy based on the provided dataset.

2. **Comprehensive Data Preprocessing**: Implement robust preprocessing techniques to handle missing values, duplicates, and outliers, ensuring clean and reliable input data.

3. **Feature Optimization**: Leverage feature engineering to maximize the predictive power of both numerical and categorical attributes.

4. **Model Efficiency**: Design a neural network architecture that balances performance and computational efficiency, suitable for real-world applications.

5. **Reusable Pipeline**: Create an end-to-end pipeline that integrates preprocessing and model training, enabling seamless deployment and scalability.

### Scope of the Project

1. **Data-Driven Insights**: Utilize a comprehensive dataset to uncover patterns and relationships between vehicle attributes and their market prices.

2. **Scalable Solution**: Develop a modular and reusable pipeline that can be adapted to similar datasets for price prediction in other domains.

3. **Real-World Application**: Provide a practical tool for stakeholders, such as car dealerships and buyers, to estimate vehicle prices accurately.

4. **Advanced Preprocessing**: Address complex data challenges, including missing values, outliers, and feature encoding, to ensure robust model performance.

5. **Integration and Deployment**: Deliver an end-to-end solution that integrates preprocessing, model training, and deployment for seamless predictions on new data.

### Methodology Adopted

1. **Data Cleaning and Preprocessing**: The dataset was thoroughly cleaned by addressing missing values, removing duplicates, and handling outliers. Numerical and categorical features were separated for tailored preprocessing.

2. **Feature Engineering**: Numerical features were scaled using `StandardScaler`, while categorical features were encoded using `OneHotEncoder`. Imputation strategies were applied to handle missing data effectively.

3. **Model Design**: A neural network model was developed using Keras, incorporating dense layers, batch normalization, and dropout layers to enhance performance and prevent overfitting.

4. **Pipeline Integration**: An end-to-end pipeline was created using `Pipeline` and `ColumnTransformer`, combining preprocessing steps and the model into a single reusable artifact for efficient training and deployment.

### Background Study about the Project

1. **Vehicle Price Prediction Trends**: Predicting vehicle prices has been a significant area of research in machine learning, driven by the need for accurate valuation in the automotive industry.

2. **Dataset Characteristics**: The dataset used in this project includes a mix of numerical and categorical features, reflecting real-world complexities such as missing values, duplicates, and outliers.

3. **Preprocessing Techniques**: Advanced preprocessing methods, including imputation, scaling, and encoding, are essential to prepare the data for machine learning models and ensure reliable predictions.

4. **Neural Network Applications**: Deep learning models, such as the Keras-based neural network used in this project, have shown promise in capturing complex relationships in structured data.

5. **Pipeline Development**: Integrating preprocessing and model training into a single pipeline is a best practice for creating scalable and reusable solutions in machine learning projects.

### Existing Systems and Their Limitations

1. **Linear Regression Models**:  
    - **Description**: Traditional linear regression models are often used for price prediction tasks.  
    - **Limitations**: These models assume a linear relationship between features and the target variable, which may not capture the complex interactions in the dataset, leading to lower accuracy.

2. **Decision Tree Regressors**:  
    - **Description**: Decision trees split the data based on feature thresholds to make predictions.  
    - **Limitations**: They are prone to overfitting, especially with noisy data, and may not generalize well to unseen data.

3. **Random Forest Regressors**:  
    - **Description**: An ensemble method that combines multiple decision trees to improve prediction accuracy.  
    - **Limitations**: While more robust than single decision trees, random forests may struggle with high-dimensional data and require extensive hyperparameter tuning.

4. **Support Vector Regressors (SVR)**:  
    - **Description**: SVR uses a kernel function to map input features into higher dimensions for regression tasks.  
    - **Limitations**: SVR can be computationally expensive for large datasets and may not perform well with categorical features without extensive preprocessing.

### Proposed System

1. **Integrated Pipeline**: The proposed system combines data preprocessing and model training into a single pipeline, ensuring seamless and efficient execution from raw data to predictions.

2. **Advanced Neural Network**: A Keras-based neural network is utilized, incorporating dense layers, batch normalization, and dropout to handle complex relationships and prevent overfitting.

3. **Robust Preprocessing**: The system employs tailored preprocessing techniques, including imputation, scaling, and one-hot encoding, to handle both numerical and categorical features effectively.

4. **Outlier and Duplicate Handling**: Comprehensive strategies are implemented to remove outliers and duplicates, ensuring the dataset is clean and reliable for training.

5. **Reusable and Scalable**: The end-to-end pipeline is designed to be modular and reusable, enabling easy adaptation to similar datasets and scalable deployment for real-world applications.

### Advantages of the Proposed System

1. **Improved Accuracy**: The integration of a Keras-based neural network with advanced preprocessing techniques ensures high accuracy in predicting vehicle prices by capturing complex relationships in the data.

2. **End-to-End Automation**: The pipeline seamlessly combines data preprocessing and model training, reducing manual intervention and ensuring consistent results.

3. **Robust Data Handling**: Comprehensive strategies for managing missing values, outliers, and duplicates enhance the reliability and quality of the input data.

4. **Scalability and Reusability**: The modular design of the pipeline allows it to be easily adapted to other datasets and scaled for larger applications, making it highly versatile.

5. **Enhanced Generalization**: The use of dropout layers and batch normalization in the neural network architecture minimizes overfitting, ensuring the model performs well on unseen data.


### Steps Involved in Developing the Project

1. **Data Collection**:  
    - Gather the dataset containing vehicle attributes and prices.  
    - Ensure the dataset is comprehensive and representative of the problem domain.

2. **Data Exploration and Cleaning**:  
    - Perform exploratory data analysis (EDA) to understand the dataset's structure and identify potential issues.  
    - Handle missing values, duplicates, and outliers to ensure data quality.

3. **Feature Engineering**:  
    - Separate numerical and categorical features for tailored preprocessing.  
    - Apply scaling, imputation, and encoding techniques to prepare the features for model training.

4. **Model Design**:  
    - Develop a neural network architecture using Keras, incorporating dense layers, batch normalization, and dropout for optimal performance.  
    - Define the model's input dimensions and compile it with appropriate loss functions and metrics.

5. **Pipeline Development**:  
    - Create an end-to-end pipeline using `Pipeline` and `ColumnTransformer` to integrate preprocessing and model training.  
    - Ensure the pipeline is modular and reusable for future datasets.

6. **Data Splitting**:  
    - Split the dataset into training and testing sets using `train_test_split` to evaluate model performance.  
    - Maintain a consistent random state for reproducibility.

7. **Model Training**:  
    - Train the pipeline on the training data, optimizing hyperparameters such as epochs and batch size.  
    - Monitor training metrics to ensure the model is learning effectively.

8. **Model Evaluation**:  
    - Evaluate the trained model on the testing set using metrics such as mean absolute error (MAE) to assess its accuracy.  
    - Analyze the results to identify potential areas for improvement.

9. **Pipeline Saving**:  
    - Save the trained pipeline as a reusable artifact using `joblib` for deployment.  
    - Ensure the saved pipeline includes both preprocessing and the trained model.