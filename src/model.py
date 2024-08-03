from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_evaluate_model(df):
    from src.data_preprocessing import split_and_scale_data
    
    x_train, x_test, y_train, y_test = split_and_scale_data(df)
    
    # Train Logistic Regression model
    lg = LogisticRegression()
    lg.fit(x_train, y_train)
    
    # Evaluate model
    y_pred = lg.predict(x_test)
    print(classification_report(y_test, y_pred))
    
    return lg, x_test, y_test, y_pred

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 5))
    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=['Not Attrite', 'Attrite'], yticklabels=['Not Attrite', 'Attrite'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()