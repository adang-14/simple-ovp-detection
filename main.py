import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder


# Load the dataset
data = pd.read_csv('healthcare_insurance_claims.csv')

# Encode categorical variables
le_gender = LabelEncoder()
data['patient_gender'] = le_gender.fit_transform(data['patient_gender'])

# Diagnosis codes

diag_data = data[['claim_id','code1','code2','code3']]
diag_data['diagnosis_codes'] = diag_data[['code1', 'code2', 'code3']].values.tolist()
diag_data = diag_data[['claim_id','diagnosis_codes']].apply(feature_engineering, axis=1)

# Join back
data.drop(columns=['code1','code2','code3'])
pd.merge(data, diag_data, on='claim_id')

# Separate features and target variable
X = data.drop(columns=[['fraud_label','claim_id']])
y = data['fraud_label']

# Create a stratified k-fold cross-validator
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=14)

# Define the XGBoost classifier
xgb_clf = xgb.XGBClassifier(random_state=14)

# Define the grid search parameters
param_grid = {
    'n_estimators': [50, 100, 200, 500, 1000],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10]
}

# Create a GridSearchCV object
grid_search = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    scoring='f1',
    cv=kfold,
    verbose=1,
    n_jobs=-1
)

# Fit the GridSearchCV object to the data
grid_search.fit(X, y)
