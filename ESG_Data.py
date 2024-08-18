#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


# In[2]:


data = pd.read_csv('C:/Users/Akul/Desktop/esgdatanifty50.csv')


# In[3]:


def adjust_features(df):
    # Amplify factor
    amplification_factor = 5
    
    df['adjusted_controversy_score'] = -df['controversy_score'] * amplification_factor
    
    df['adjusted_esg_risk_level'] = df['esg_risk_level'].apply(
        lambda x: -1 * amplification_factor if x == 'Low' else (1 * amplification_factor if x == 'High' else 0)
    )
    
    df['adjusted_esg_risk_exposure'] = df['esg_risk_exposure'].apply(
        lambda x: -1 * amplification_factor if x == 'Low' else (1 * amplification_factor if x == 'Significant' else -0)
    )
    
    df['adjusted_esg_risk_management'] = df['esg_risk_management'].apply(
        lambda x: -1 * amplification_factor if x == 'Strong' else (1 * amplification_factor if x == 'Weak' else -0)
    )
    
    return df


# In[4]:


data = adjust_features(data)
data


# In[5]:


# Selecting features and target variable
numerical_features_to_normalize = ['esg_score_2019', 'esg_score_2020', 'esg_score_2021']
adjusted_features = ['adjusted_controversy_score', 'adjusted_esg_risk_level', 'adjusted_esg_risk_exposure', 'adjusted_esg_risk_management']
numerical_features_no_scaling = []


# In[6]:


target = data['esg_risk_score_2024']


# In[7]:


# Preprocessing for numerical data
numerical_transformer = ColumnTransformer(
    transformers=[
        ('num_normalize', StandardScaler(), numerical_features_to_normalize),
        ('num_passthrough', 'passthrough', numerical_features_no_scaling)
    ]
)


# In[8]:


categorical_transformer = OneHotEncoder(handle_unknown='ignore')


# In[9]:


# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features_to_normalize),
        ('adj', 'passthrough', adjusted_features)
    ])


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(data[numerical_features_to_normalize + adjusted_features], target, test_size=0.2, random_state=42)


# In[11]:


#model = XGBRegressor(n_estimators=500, random_state=42, booster='gblinear', eta = '0.1', reg_lambda = ' 1',
 #                    reg_alpha = '1')
    
model = RandomForestRegressor(
    n_estimators=100,             # Increase the number of trees
    max_depth=8,                 # Increase the depth of the trees
    min_samples_split=2,          # Require more samples to split a node
    min_samples_leaf=1,           # Require more samples at a leaf node
    max_features='sqrt',          # Use a subset of features for splitting
    bootstrap=True,               # Use bootstrap sampling
    oob_score=True,               # Use out-of-bag samples to estimate R²
    random_state=42,
    n_jobs=-1,                    # Use all available cores
    verbose=1 
)


# In[12]:


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# Train the model
pipeline.fit(X_train, y_train)


# In[13]:


# Step 6: Validation
predictions = pipeline.predict(X_test)
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R² Score: {r2}')


# In[14]:


# Predict future ESG scores
future_predictions = pipeline.predict(data[numerical_features_to_normalize + adjusted_features])
data['predicted_future_esg_score'] = future_predictions

print(data[['Symbol','company', 'esg_risk_score_2024', 'predicted_future_esg_score']])


# In[15]:


# Plotting predicted score vs 2024 ESG risk score for each company
plt.figure(figsize=(10, 6))

# Bar width
bar_width = 0.35

# Positions of the bars on the x-axis
index = range(len(data))

# Plotting the bars
plt.bar(index, data['esg_risk_score_2024'], bar_width, label='ESG Risk Score 2024')
plt.bar([p + bar_width for p in index], data['predicted_future_esg_score'], bar_width, label='Predicted Future ESG Score')

# Adding the company labels
plt.xlabel('Company')
plt.ylabel('Score')
plt.title('Comparison of ESG Risk Score 2024 and Predicted Future ESG Score')
plt.xticks([p + bar_width / 2 for p in index], data['company'])
plt.legend()

plt.show()


# In[16]:


newd = data[['Symbol', 'company', 'Sector', 'Industry', 'Description', 'esg_risk_score_2024', 'predicted_future_esg_score', 'esg_risk_exposure', 'esg_risk_management', 'esg_risk_level']]


# In[18]:


# Merge with another CSV
additional_data = pd.read_csv('C:\\Users\\Akul\\Downloads\\lol.csv')
newd = newd.merge(additional_data, on='Symbol', how='left')

newd

# Convert to JSON
#newd.to_json('final_data.json', orient='records', lines=True)


# In[22]:


newd.to_csv('C:\\Users\\Akul\\Downloads\\final_data.csv', index=False)


# In[ ]:




