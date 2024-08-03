from src.data_preprocessing import load_and_preprocess_data
from src.model import train_and_evaluate_model
from src.visualization import plot_confusion_matrix

def main():
    # Load and preprocess data
    df = load_and_preprocess_data('data/HR_Employee_Attrition.xlsx')
    
    # Train and evaluate model
    model, X_test, y_test, y_pred = train_and_evaluate_model(df)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)

if __name__ == "__main__":
    main()