from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load pre-trained assets
data = pd.read_csv("recipe_dataset.csv")
scaler = joblib.load("scaler.pkl")
vectorizer = joblib.load("vectorizer.pkl")
model = load_model("best_model.h5", compile=False)  # Load FFNN model correctly

# Preprocess full dataset for inference
X_ingredients = vectorizer.transform(data['ingredients_list'])
X_numerical = scaler.transform(data[['calories', 'fat', 'carbohydrates', 'protein', 'cholesterol', 'sodium', 'fiber']])
X_combined = np.hstack([X_numerical, X_ingredients.toarray()])
recipe_embeddings = model.predict(X_combined)

# Recommendation function
def recommend(input_features):
    nutrition = scaler.transform([input_features[:7]])
    ingredients = vectorizer.transform([input_features[7]])
    input_combined = np.hstack([nutrition, ingredients.toarray()])

    embedding = model.predict(input_combined)
    sims = cosine_similarity(embedding, recipe_embeddings).flatten()
    indices = sims.argsort()[-5:][::-1]
    result = data.iloc[indices]

    return result[['recipe_name', 'ingredients_list', 'image_url']].to_dict(orient='records')

# Truncate helper
def truncate(text, length):
    return text[:length] + "..." if len(text) > length else text

@app.route("/", methods=['GET', 'POST'])
def home():
        return render_template('open.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    form_data = None  # Initialize form_data variable to None
    recommendations = []

    if request.method == 'POST':
        try:
            # Collect features from the form
            features = [
                float(request.form['calories']),
                float(request.form['fat']),
                float(request.form['carbohydrates']),
                float(request.form['protein']),
                float(request.form['cholesterol']),
                float(request.form['sodium']),
                float(request.form['fiber']),
                request.form['ingredients']
            ]

            # Store form data to pass it to the template
            form_data = {
                'calories': features[0],
                'fat': features[1],
                'carbohydrates': features[2],
                'protein': features[3],
                'cholesterol': features[4],
                'sodium': features[5],
                'fiber': features[6],
                'ingredients': features[7]
            }

            # Get recommendations based on the input features
            recommendations = recommend(features)

        except Exception as e:
            return f"Error: {e}"

    return render_template('index.html', recommendations=recommendations, form_data=form_data, truncate=truncate)


if __name__ == '__main__':
    app.run(debug=True)
