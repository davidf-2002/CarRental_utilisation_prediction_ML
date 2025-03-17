from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

class ModelTrainer:
    def __init__(self, model_path='models'):
        self.model = RandomForestClassifier(random_state=42)
        self.model_path = model_path
        
    def train(self, X_train, y_train):
        """Train the model"""
        self.model.fit(X_train, y_train)
        
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        
        return {
            'accuracy': accuracy,
            'classification_report': report
        }
    
    def save_model(self, filename='model.joblib'):
        """Save the trained model"""
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        
        path = os.path.join(self.model_path, filename)
        joblib.dump(self.model, path)
        
    def load_model(self, filename='model.joblib'):
        """Load a trained model"""
        path = os.path.join(self.model_path, filename)
        self.model = joblib.load(path)
