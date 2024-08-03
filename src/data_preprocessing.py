import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    # Load dataset
    df = pd.read_excel(file_path)
    
    # Drop unnecessary columns
    df = df.drop(['EmployeeNumber', 'Over18', 'StandardHours'], axis=1)
    
    # Create dummy variables
    to_get_dummies_for = ['BusinessTravel', 'Department', 'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender', 'JobInvolvement', 'JobLevel', 'JobRole', 'MaritalStatus']
    df = pd.get_dummies(data=df, columns=to_get_dummies_for, drop_first=True)
    
    # Map categorical variables
    dict_OverTime = {'Yes': 1, 'No': 0}
    dict_attrition = {'Yes': 1, 'No': 0}
    df['OverTime'] = df.OverTime.map(dict_OverTime)
    df['Attrition'] = df.Attrition.map(dict_attrition)
    
    return df

def split_and_scale_data(df):
    # Separate target variable and other variables
    Y = df.Attrition
    X = df.drop(columns=['Attrition'])
    
    # Scale the data
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=1, stratify=Y)
    
    return x_train, x_test, y_train, y_test