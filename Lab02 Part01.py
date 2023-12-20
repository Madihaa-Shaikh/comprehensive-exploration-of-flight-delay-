#!/usr/bin/env python
# coding: utf-8

# In[56]:


#read all data and print all features
import pandas as pd
flight_data = pd.read_csv('aa-delays-2023.csv')
flight_data


# In[57]:


# 3. Print All Features
print("All Features (Column Names):")
print(flight_data.columns)


# In[58]:


# 5 How can we determine how strong is the influence of WEATHER_DELAY on ARR_DELAY?
correlation = flight_data['WEATHER_DELAY'].corr(flight_data['ARR_DELAY'])
print(f"Correlation between WEATHER_DELAY and ARR_DELAY: {correlation}")


# In[59]:


#correlation
import seaborn as sns
import matplotlib.pyplot as plt

if 'WEATHER_DELAY' in flight_data.columns and 'ARR_DELAY' in flight_data.columns:
    # Scatter plot with regression line
    sns.regplot(x='WEATHER_DELAY', y='ARR_DELAY', data=flight_data)
    
    # Show the plot
    plt.show()
else:
    print("Columns 'WEATHER_DELAY' or 'ARR_DELAY' not found in the dataset. Please verify your data.")


# In[11]:


#If the scatter plot with the regression line appears as a straight line, it suggests a linear relationship between the two variables. A linear relationship means that as one variable (e.g., WEATHER_DELAY) increases or decreases, the other variable (ARR_DELAY) changes proportionally.


# In[60]:


# 6 Delete ARR_DELAY andString data / Assuming 'ARR_DELAY' represents the total delay duration
if 'ARR_DELAY' in flight_data.columns:
    flight_data = flight_data.drop(['ARR_DELAY'], axis=1)
    print("'ARR_DELAY' column dropped.")
else:
    print("Column 'ARR_DELAY' not found in the dataset. No action taken.")

# Drop other unwanted string columns
columns_to_drop = ['ORIGIN', 'DEST', 'Unnamed: 27','OP_CARRIER']
flight_data = flight_data.drop(columns_to_drop, axis=1)


# In[61]:


# 7 Set thetargetofa delay>15 minutesto1 otherwise to/  Assuming a new binary column 'DELAY_TARGET' to represent delays > 15 minutes
target_feature = 'DEP_DELAY'  # Replace with the actual column name representing delay
threshold = 15  # Set the threshold for delay in minutes

if target_feature in flight_data.columns:
    flight_data['DELAY_TARGET'] = (flight_data[target_feature] > threshold).astype(int)
    print(f"'DELAY_TARGET' column created.")
else:
    print(f"Column '{target_feature}' not found in the dataset. Please verify your data.")


# In[62]:


# 4 The goal is to reduce the cost of flight delay. Which target feature do we choose and why?
# Assuming 'ARR_DELAY' represents the total delay duration
target_feature = 'ARR_DELAY'
print(f"Chosen Target Feature: {target_feature}")

flight_data


# In[63]:


if 'FL_DATE' in flight_data.columns:
    flight_data = flight_data.drop(['FL_DATE'], axis=1)
    print("'FL_DATE' column dropped.")
flight_data


# In[67]:


# Assuming 'DELAY_TARGET' is the target feature
target_feature = 'DELAY_TARGET'
if target_feature not in flight_data.columns:
    print(f"Error: Target feature '{target_feature}' not found in the DataFrame.")
else:
    # Choose two specific columns for correlation
    columns_of_interest = ['WEATHER_DELAY', 'DEP_DELAY']  # Replace with your actual column names

    # Calculate Pearson correlation coefficients with 'DELAY_TARGET' for the chosen columns
    correlations_with_target = flight_data[columns_of_interest + [target_feature]].corr()[target_feature]

    # Print correlations
    print(f"Linear Correlations of '{target_feature}' with {columns_of_interest}:")
    print(correlations_with_target)


# In[64]:


flight_data


# In[ ]:





# In[ ]:




