import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

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

def train_and_save_model():
    # Ensure the script is run from the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load the dataset
    df = pd.read_csv(os.path.join(script_dir, "intel.csv"))
    
    # Prepare features and target
    X = df[['Age', 'Last_Test_Score', 'Knowledge_Level', 'Learning_Speed']].copy()
    y = df['Next_Level']
    
    # Encode categorical variables
    le_knowledge = LabelEncoder()
    le_learning = LabelEncoder()
    le_target = LabelEncoder()
    
    # Properly transform categorical columns
    X['Knowledge_Level'] = le_knowledge.fit_transform(X['Knowledge_Level'])
    X['Learning_Speed'] = le_learning.fit_transform(X['Learning_Speed'])
    y = le_target.fit_transform(y)
    
    # Normalize numerical features
    scaler = StandardScaler()
    X[['Age', 'Last_Test_Score']] = scaler.fit_transform(X[['Age', 'Last_Test_Score']])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create and train the model
    model = EqualFeatureImportanceClassifier(
        max_depth=5,  # Prevent overfitting
        random_state=42,
        class_weight='balanced'
    )
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Save the model and encoders
    joblib.dump(model, os.path.join(script_dir, "student_performance_model.pkl"))
    joblib.dump({
        'knowledge_encoder': le_knowledge,
        'learning_encoder': le_learning,
        'target_encoder': le_target,
        'scaler': scaler
    }, os.path.join(script_dir, "label_encoders.pkl"))
    
    print("Model and encoders saved successfully!")
    
    return model, {
        'knowledge_encoder': le_knowledge,
        'learning_encoder': le_learning,
        'target_encoder': le_target,
        'scaler': scaler
    }

# Run the training and save the model
if __name__ == "__main__":
    train_and_save_model()