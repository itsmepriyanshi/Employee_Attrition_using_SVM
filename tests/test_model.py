import pytest
from src.model import train_and_evaluate_model
from src.data_preprocessing import load_and_preprocess_data

def test_train_and_evaluate_model():
    df = load_and_preprocess_data('data/HR_Employee_Attrition.xlsx')
    model, X_test, y_test, y_pred = train_and_evaluate_model(df)
    assert model is not None
    assert X_test.shape[0] == y_test.shape[0]
    assert y_pred.shape[0] == y_test.shape[0]