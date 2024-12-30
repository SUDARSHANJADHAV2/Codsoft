import pandas as pd
import numpy as np
import ast
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

def load_dataset(file_path):
    """
    Load dataset from a specified file path
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        # Read the CSV
        data = pd.read_csv(file_path)
        
        # Convert genres from string representation to list
        data['genres'] = data['genres'].apply(ast.literal_eval)
        
        # Drop rows with missing values
        data.dropna(subset=['plot'], inplace=True)
        
        return data
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

def prepare_data(data):
    """Prepare data for multi-label classification"""
    # Multi-label binarization
    mlb = MultiLabelBinarizer()
    data_genres = mlb.fit_transform(data['genres'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data['plot'], data_genres, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, mlb

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """Train and evaluate the genre classification model"""
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train model
    model = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42))
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return model, vectorizer

def save_model_components(model, vectorizer, mlb):
    """Save trained model components"""
    try:
        joblib.dump(model, 'genre_classifier_model.pkl')
        joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
        joblib.dump(mlb, 'mlb.pkl')
        print("Model components saved successfully.")
    except Exception as e:
        print(f"Error saving model components: {e}")

def create_genre_predictor(vectorizer, model, mlb):
    """Create a function to predict genres"""
    def predict_genres(plot):
        plot_tfidf = vectorizer.transform([plot])
        prediction = model.predict(plot_tfidf)
        genres = mlb.inverse_transform(prediction)
        return genres
    return predict_genres

def main():
    # Path to the dataset (modify if needed)
    dataset_path = 'genre-classification-dataset.csv'
    
    try:
        # Load dataset
        data = load_dataset(dataset_path)
        
        # Prepare data
        X_train, X_test, y_train, y_test, mlb = prepare_data(data)
        
        # Train and evaluate model
        model, vectorizer = train_and_evaluate_model(X_train, X_test, y_train, y_test)
        
        # Save model components
        save_model_components(model, vectorizer, mlb)
        
        # Create genre predictor
        predict_genres = create_genre_predictor(vectorizer, model, mlb)
        
        # Print available genres
        print("\nAvailable Genres:")
        print(mlb.classes_)
        
        # Interactive prediction loop
        while True:
            user_input = input("\nEnter a movie plot summary (or type 'exit' to quit): ")
            if user_input.lower() == 'exit':
                print("Exiting the genre prediction tool. Goodbye!")
                break
            
            if not user_input.strip():
                print("Please enter a valid plot summary.")
                continue
            
            predicted_genres = predict_genres(user_input)
            print(f"Predicted Genres: {predicted_genres}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()