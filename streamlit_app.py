import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier

class EqualFeatureImportanceClassifier(DecisionTreeClassifier):
    def __init__(self, **kwargs):
        # Call the parent class constructor with all provided arguments
        super().__init__(**kwargs)
        self._feature_importances = None
    
    def fit(self, X, y, sample_weight=None):
        # Fit the model using parent class method
        super().fit(X, y, sample_weight)
        
        # Set equal feature importances
        n_features = X.shape[1]
        self._feature_importances = np.ones(n_features) / n_features
        
        return self
    
    @property
    def feature_importances_(self):
        return self._feature_importances

def load_model_and_encoders():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # Load the saved model
        model_path = os.path.join(script_dir, "student_performance_model.pkl")
        encoders_path = os.path.join(script_dir, "label_encoders.pkl")
        
        model = joblib.load(model_path)
        encoders = joblib.load(encoders_path)
        
        return model, encoders
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def predict_next_level(model, encoders, input_data):
    # Get the label encoders and scaler
    knowledge_encoder = encoders['knowledge_encoder']
    learning_encoder = encoders['learning_encoder']
    target_encoder = encoders['target_encoder']
    scaler = encoders['scaler']
    
    # Prepare the input data
    input_df = pd.DataFrame([input_data])
    
    # Encode categorical variables
    input_df['Knowledge_Level'] = knowledge_encoder.transform(input_df['Knowledge_Level'])
    input_df['Learning_Speed'] = learning_encoder.transform(input_df['Learning_Speed'])
    
    # Normalize numerical features 
    input_df[['Age', 'Last_Test_Score']] = scaler.transform(input_df[['Age', 'Last_Test_Score']])
    
    # Make prediction
    prediction = model.predict(input_df)
    
    # Decode the prediction
    predicted_level = target_encoder.inverse_transform(prediction)[0]
    
    return predicted_level

def main():
    st.title("Student Performance Next Level Predictor")
    
    # Load the model and encoders
    model, encoders = load_model_and_encoders()
    
    if model is None or encoders is None:
        st.error("Could not load the model. Please run the training script first.")
        return
    
    # Get unique values for categorical features
    knowledge_levels = encoders['knowledge_encoder'].classes_
    learning_speeds = encoders['learning_encoder'].classes_
    
    # Create input fields
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, value=20)
        last_test_score = st.number_input("Last Test Score", min_value=0, max_value=100, value=50)
    
    with col2:
        knowledge_level = st.selectbox("Knowledge Level", knowledge_levels)
        learning_speed = st.selectbox("Learning Speed", learning_speeds)
    
    # Prediction button
    if st.button("Predict Next Level"):
        # Prepare input data
        input_data = {
            'Age': age,
            'Last_Test_Score': last_test_score,
            'Knowledge_Level': knowledge_level,
            'Learning_Speed': learning_speed
        }
        
        # Make prediction
        predicted_level = predict_next_level(model, encoders, input_data)
        
        # Display prediction
        st.success(f"Predicted Next Level: {predicted_level}")
        
        # Optional: Add some explanation or additional insights
        st.info("This prediction is based on the machine learning model trained on historical student performance data.")

# Run the app
if __name__ == "__main__":
    main()