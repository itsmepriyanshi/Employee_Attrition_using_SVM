import pytest
import pandas as pd
from src.data_preprocessing import load_and_preprocess_data, split_and_scale_data

def test_load_and_preprocess_data():
    df = load_and_preprocess_data('data/HR_Employee_Attrition.xlsx')
    assert isinstance(df, pd.DataFrame)
    assert 'Attrition' in df.columns

def test_split_and_scale_data():
    df = load_and_preprocess_data('data/HR_Employee_Attrition.xlsx')
    x_train, x_test, y_train, y_test = split_and_scale_data(df)
    assert x_train.shape[0] == y_train.shape[0]
    assert x_test.shape[0] == y_test.shape[0]