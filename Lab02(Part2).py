#!/usr/bin/env python
# coding: utf-8

# In[93]:


#read all data and print all features
import pandas as pd
flight_data = pd.read_csv('aa-delays-2023(02).csv')
# 7 Set thetargetofa delay>15 minutesto1 otherwise to/  Assuming a new binary column 'DELAY_TARGET' to represent delays > 15 minutes
target_feature = 'DEP_DELAY'  # Replace with the actual column name representing delay
threshold = 15  # Set the threshold for delay in minutes

if target_feature in flight_data.columns:
    flight_data['DELAY_TARGET'] = (flight_data[target_feature] > threshold).astype(int)
    print(f"'DELAY_TARGET' column created.")
else:
    print(f"Column '{target_feature}' not found in the dataset. Please verify your data.")

    
flight_data


# In[94]:


#10. Test Normal Distribution of ARR_DELAY (using only the first 100 data sets) and Print the Histogram:

from scipy.stats import kstest
import matplotlib.pyplot as plt

# Assuming 'ARR_DELAY' is the feature of interest
arr_delay_sample = flight_data['ARR_DELAY'][:100]

# Kolmogorov-Smirnov test for normality
ks_statistic, ks_p_value = kstest(arr_delay_sample, 'norm')

# Print KS test result
print(f"KS Statistic: {ks_statistic}, p-value: {ks_p_value}")

# Plot histogram
plt.hist(arr_delay_sample, bins='auto', alpha=0.7, color='blue', edgecolor='black')
plt.title('Histogram of ARR_DELAY')
plt.xlabel('ARR_DELAY')
plt.ylabel('Frequency')
plt.show()


# In[95]:


#11. Perform a Nonlinear Transformation and Check for Normal Distribution:You can try different transformations (e.g., logarithmic, square root) and check for normality:

import numpy as np
from scipy.stats import kstest
import matplotlib.pyplot as plt


# Example: Logarithmic transformation
arr_delay_transformed = np.log1p(arr_delay_sample)

# Perform KS test on the transformed data
ks_statistic_transformed, ks_p_value_transformed = kstest(arr_delay_transformed, 'norm')

# Print KS test result for the transformed data
print(f"KS Statistic (Transformed): {ks_statistic_transformed}, p-value: {ks_p_value_transformed}")

# Plot histogram of transformed data
plt.hist(arr_delay_transformed, bins='auto', alpha=0.7, color='green', edgecolor='black')
plt.title('Histogram of Transformed ARR_DELAY')
plt.xlabel('Transformed ARR_DELAY')
plt.ylabel('Frequency')
plt.show()


# In[96]:


print(flight_data.columns)



# In[97]:


import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Assume you have a DataFrame named 'flight_data'

# Selecting relevant features
selected_features = [
    'CRS_DEP_TIME', 'DEP_DELAY', 'TAXI_OUT', 'WHEELS_OFF',
    'WHEELS_ON', 'TAXI_IN', 'CRS_ARR_TIME', 'CRS_ELAPSED_TIME',
    'ACTUAL_ELAPSED_TIME', 'AIR_TIME', 'DISTANCE', 'CARRIER_DELAY',
    'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY'
]

# Creating a new DataFrame with only the selected features and target variable
selected_data = flight_data[selected_features + ['DELAY_TARGET']]

# Handling missing values
selected_data.fillna(0, inplace=True)

# Splitting the data into features (X) and target variable (y)
X = selected_data.drop('DELAY_TARGET', axis=1)
y = selected_data['DELAY_TARGET']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a decision tree regressor
dt_regressor = DecisionTreeRegressor()

# Fit the model
dt_regressor.fit(X_train, y_train)

# Make predictions
y_pred = dt_regressor.predict(X_test)

# Print mean squared error and R2 score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")


# In[ ]:





# In[ ]:




