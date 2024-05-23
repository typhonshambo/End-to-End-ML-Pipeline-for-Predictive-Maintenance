import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

class Evaluator:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = self.model.predict(self.X_test)
    
    def compute_metrics(self):
        """Compute and return evaluation metrics."""
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred, zero_division=1)
        recall = recall_score(self.y_test, self.y_pred, zero_division=1)
        f1 = f1_score(self.y_test, self.y_pred, zero_division=1)
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        return metrics
    
    def print_classification_report(self):
        """Print classification report."""
        report = classification_report(self.y_test, self.y_pred, zero_division=1)
        print(report)
    
    def plot_confusion_matrix(self):
        """Plot the confusion matrix."""
        cm = confusion_matrix(self.y_test, self.y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

# Example usage:
# if __name__ == "__main__":
#     df = pd.read_csv('data/processed/preprocessed_data.csv')
#     trainer = ModelTrainer(df)
#     X_train, X_test, y_train, y_test = trainer.prepare_data(target_column='failure')
#     model = trainer.train_model
