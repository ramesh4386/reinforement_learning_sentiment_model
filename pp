#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/gist/tyty9798123/f302c49c7154e98c52067822e08ede89/experiment4.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


# from google.colab import drive
# drive.mount('/content/drive')


# # Import packages

# In[2]:


import numpy as np
import pandas as pd
import os

import time
import datetime
import glob

import warnings
warnings.filterwarnings("ignore")

from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (6.4, 4.8)
plt.rcParams["figure.dpi"] = 300
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import ClusterCentroids
from sklearn.ensemble import BaggingClassifier

import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# # Data Preprocessing

# ## Change Time Zone

# In[3]:


# !date -R
# os.environ["TZ"] = "America/New_York"
# time.tzset()
# !date -R


# ## Load Data

# In[4]:


# dir = './drive/MyDrive/stocktwits_dataset'
# symbol = 'SPY'
# my_dir = os.path.join(dir, symbol)


# In[5]:


#saved_path = os.path.join(my_dir, symbol+'_Finbert_processed.csv') # read machine-labeled and user-labeled
df = pd.read_csv("SPY_concat_files (3).csv")


# In[6]:


df.shape


# In[7]:


df=df.head(2000)


# ## Load Historical Stock Data

# In[8]:


#fin_dir = os.path.join('./drive/MyDrive/stocktwits_dataset', 'Financial', symbol+'.csv')
fin_df = pd.read_csv("SPY_15Min_2008-Sep2023.csv")
fin_df.head()


# In[9]:


fin_df["Date"] = pd.to_datetime(fin_df["Time"])


# ## Load Historcial Stock Price (Intraday)

# In[10]:


os.getcwd()


# In[11]:


fifteen_mins_df = pd.read_csv("SPY_15Min_2008-Sep2023.csv")


# In[12]:


fifteen_mins_df["time"] = pd.to_datetime(fifteen_mins_df["Time"])


# In[13]:


fifteen_mins_df


# In[14]:


# # # intraday stock price (15 mins update)
# # path_15mins = os.getcwd() #'./drive/MyDrive/stocktwits_dataset/fifteen_mins/' + symbol
# # all_files = glob.glob(os.path.join(SPY_15Min_2008-Sep2023, '*.csv')
# # )
# # li = []
# # for filename in all_files:
# #     df = pd.read_csv(filename, index_col=None, header=0)
# #     li.append(df)

# # fifteen_mins_df = pd.concat(li, axis=0, ignore_index=True)
# # fifteen_mins_df = fifteen_mins_df.drop_duplicates(subset=['time'])
# # Divide DateTime
# #import Time
# Dates = []
# Times = []
# for index, row in fifteen_mins_df.iterrows():
#   Dates.append(
#       row["time"].split(" ")[0]
#   )
#   Times.append(
#       row["time"].split(" ")[1]
#   )
# fifteen_mins_df["Date"] = Dates
# fifteen_mins_df["Time"] = Times
# fifteen_mins_df = fifteen_mins_df.drop(columns=['time'])
# fifteen_mins_df.head()


# In[15]:


# Extract Date and Time from the "time" column
fifteen_mins_df["Date"] = fifteen_mins_df["time"].dt.date
fifteen_mins_df["Time"] = fifteen_mins_df["time"].dt.time

# Drop the original "time" column
fifteen_mins_df = fifteen_mins_df.drop(columns=['time'])

fifteen_mins_df.head()


# In[16]:


# Convert 'Date' column to datetime type
fifteen_mins_df['Date'] = pd.to_datetime(fifteen_mins_df['Date'])

# Example date range for filtering
start_date = '2008-05-05' 
end_date = '2009-11-12'

# Filter the data based on the date range
fifteen_mins_df = fifteen_mins_df[(fifteen_mins_df['Date'] >= start_date) & (fifteen_mins_df['Date'] <= end_date)]


# In[17]:


fifteen_mins_df.shape


# ## Add Column Open Timestamp

# In[18]:


import datetime
import time

fin_df["Date"] = fin_df["Date"].astype(str)
Open_Timestamp = []

for i in range(len(fin_df)):
    temp = time.mktime(datetime.datetime.strptime(fin_df["Date"][i], "%Y-%m-%d %H:%M:%S").timetuple()) + (9.5 * 60 * 60)
    Open_Timestamp.append(int(temp))

fin_df["Open Timestamp"] = Open_Timestamp
fin_df.head()


# In[19]:


fin_df.shape


# In[20]:


# Convert 'Date' column to datetime type
fin_df['date'] = pd.to_datetime(fin_df['Date'])

# Example date range for filtering
start_date = '2008-05-05' 
end_date = '2009-11-12'

# Filter the data based on the date range
fin_df = fin_df[(fin_df['date'] >= start_date) & (fin_df['date'] <= end_date)]
fin_df.shape


# ## Keey Only Date, Open Timestamp, Change Percent

# In[21]:


fin_tmp_df = DataFrame()
fin_tmp_df["Date"] = fin_df["Date"]
fin_tmp_df["Open Timestamp"] = fin_df["Open Timestamp"]
fin_tmp_df["Change Percent"] = fin_df["%Chg"]
fin_tmp_df.head()


# ## Get Sentiment Index

# In[22]:


def get_sentiment_index_three_class(concated, timestamp, hour_1, hour_2):
  open_timestamp = timestamp
  filter1 = concated['Timestamp'] < (open_timestamp - hour_1 *(60*60) ) * 1000 #小於Open
  filter2 = concated["Timestamp"] > (open_timestamp - hour_2 *(60*60) ) * 1000 #大於Open - 43200(12小時前)
  concated = concated.where(filter1 & filter2)
  tmp = concated.dropna(subset=["Timestamp"])
  p = tmp.where( tmp['Sentiment'] == 'positive' )
  p = p.dropna(subset=["Timestamp"])  
  neg = tmp.where( tmp['Sentiment'] == 'negative' )
  neg = neg.dropna(subset=["Timestamp"])
  neu = tmp.where( tmp['Sentiment'] == 'neutral' )
  neu = neu.dropna(subset=["Timestamp"])
  num_of_positive = p.count()["Sentiment"]
  num_of_negative = neg.count()["Sentiment"]
  num_of_neutral = neu.count()["Sentiment"]

  return (num_of_positive - num_of_negative) / (num_of_positive + num_of_negative + num_of_neutral)


# In[23]:


# concated = df.copy()
# timestamp = selected_rows["Open Timestamp"][19]
# hour_1 = 5.5
# hour_2 =9.5
# get_sentiment_index_three_class(concated, timestamp, hour_1, hour_2)


# ## Get Previous Financial Data

# In[24]:


def getPrevFinancialData(Date):
  current_index = fin_df.where(fin_df["Date"] == Date).dropna(subset=["Date"]).index[0]
  prev_index = current_index + 1
  return fin_df.iloc[prev_index]


# In[25]:


def getPrevFinancialDataWithDay(Date, num_of_days):
  current_index = fin_df.where(fin_df["Date"] == Date).dropna(subset=["Date"]).index[0]
  prev_index = current_index + num_of_days
  return fin_df.iloc[prev_index]


# ## Function of Get Percentage Change

# In[26]:


# def get_percentage_change_with_mins(date, time):
#   tmp = fifteen_mins_df[(fifteen_mins_df.date == date) & (fifteen_mins_df.time == time)]
#   tmp = tmp.dropna(subset=["time"])
#   tmp['open'] = tmp['open'].apply(lambda x: float(x))
#   tmp['high'] = tmp['high'].apply(lambda x: float(x))
#   tmp['low'] = tmp['low'].apply(lambda x: float(x))
#   tmp['close'] = tmp['last'].apply(lambda x: float(x))

#   return tmp


# In[27]:


fifteen_mins_df


# In[28]:


def get_percentage_change_with_mins(date, time):
    # Filter the DataFrame based on date and time
    tmp = fifteen_mins_df[(fifteen_mins_df["Date"] == date) & (fifteen_mins_df["Time"] == time)]

    # Check if the filtered DataFrame is empty
    if tmp.empty:
        return None  # Return None if no data is found

    # Convert relevant columns to float
    tmp['open'] = tmp['Open'].astype(float)
    tmp['high'] = tmp['High'].astype(float)
    tmp['low'] = tmp['Low'].astype(float)
    tmp['close'] = tmp['Last'].astype(float)

    return tmp


# In[29]:


# Assuming your DataFrame is named df
#fifteen_mins_df.columns = fifteen_mins_df.columns.str.lower()


# In[30]:


fifteen_mins_df = fifteen_mins_df.sort_values(by="Date")


# In[31]:


#float( get_percentage_change_with_mins('2023-09-26', "13:45:00")["close"] )


# In[32]:


# Extract numeric values from the Percentage column
fifteen_mins_df['%Chg'] = fifteen_mins_df['%Chg'].str.extract(r'([-+]?\d*\.\d+|\d+)').astype(float)
fifteen_mins_df = fifteen_mins_df.head(10000)


# In[33]:


def split(word):
    return [char for char in word]


# In[34]:


time = "1200"
num = 14
generated_file_name = f'processed_2020_with_{time}_{num}.csv'
print(generated_file_name)


# ## debugging

# In[35]:



fin_tmp_df["Time"] = pd.to_datetime(fin_tmp_df["Date"])
fin_tmp_df["Time"] = fin_tmp_df["Time"].dt.time
fin_tmp_df["Date"]= pd.to_datetime(fin_tmp_df["Date"])
fin_tmp_df["Date"] = fin_tmp_df["Date"].dt.date


# In[36]:


# Define the target time
target_time = "09:30:00"

# Convert the "Time" column to a string type
fin_tmp_df["Time"] = fin_tmp_df["Time"].astype(str)

# Filter the DataFrame to select rows with the target time
selected_rows = fin_tmp_df[fin_tmp_df["Time"].str.contains(target_time)]

# Print the selected rows
selected_rows.head()


# In[37]:


fin_tmp_df = selected_rows.copy()


# In[38]:


fin_tmp_df


# In[39]:


#fifteen_mins_df
# Convert the "Time" column to a string type
fifteen_mins_df["Time"] = fifteen_mins_df["Time"].astype(str)
fifteen_mins_df["Date"] = fifteen_mins_df["Date"].astype(str)


# In[40]:


get_percentage_change_with_mins("2008-05-06", "09:30:00")


# In[41]:


result = get_percentage_change_with_mins("2008-05-06", "09:30:00")


# In[42]:


float(result["close"])


# In[43]:


# Attempt to get percentage change for "2008-05-06" and "09:30:00"
result = get_percentage_change_with_mins("2008-05-06", "09:30:00")

# Check if the result is None
if result is not None:
    try:
        open = float(result['close'])
        time = "09:30:00"  # This is the specified time
        result = get_percentage_change_with_mins("2008-05-06", time)
        print(result)
    except:
        print("Error occurred while processing the data.")
else:
    print("No data found for '2008-05-06' and '09:30:00'.")


# In[44]:


df


# In[45]:



for index, row in fin_tmp_df.iterrows():
    timestamp = row['Open Timestamp']
#     print(row["Date"])
#     print(index + 1)
    
    result = get_percentage_change_with_mins(row["Date"], "09:30:00")
    if result is None:
#         print("Skipping row due to missing data")
        continue

    try:
        open = float(result['close'])
        time = row["Time"]
        time_str = f"{time.split(':')[0]}{time.split(':')[1]}:00"
        result = get_percentage_change_with_mins(row["Date"], time_str)
        if result is None:
            print("Skipping row due to missing data")
            continue
        close = float(result['close'])
    except Exception as e:
        print(f"Error: {e}")
        break
    
    temp = pd.Timestamp(row["Date"])
    day_of_the_week.append(int(temp.dayofweek + 1))
    dates.append(row["Date"])


# 
# ## Generate the data Before training.

# In[46]:


# Convert "Time" column to time format
fifteen_mins_df["Time"] = pd.to_datetime(fifteen_mins_df["Time"], format="%H:%M:%S").dt.time

# Convert "Date" column to date format
fifteen_mins_df["Date"] = pd.to_datetime(fifteen_mins_df["Date"], format="%Y-%m-%d").dt.date


# In[47]:


fifteen_mins_df


# In[48]:


second_df=df.copy()
# Preprocessing  
fifteen_mins_df['day_of_the_week'] = pd.to_datetime(fifteen_mins_df['Date']).dt.dayofweek + 1  
  
dates = []  
day_of_the_week = []  
prev_open = []  
prev_close = []  
prev_high = []  
prev_low = []  
prev_hlpct = []  
prev_volume = []  
prev_change_percent = []  
prev_vwap = []  
prev_trade_value = []  
percentage_change = []  
  
# Iterate through the rows of the fifteen_mins_df DataFrame  
for index, row in fifteen_mins_df.iterrows():  
    # Append the values to the corresponding lists  
    dates.append(row["Date"])  
    day_of_the_week.append(row["day_of_the_week"])  
    prev_open.append(row["Open"])  
    prev_close.append(row["Last"])  
    prev_high.append(row["High"])  
    prev_low.append(row["Low"])  
    prev_hlpct.append((row["High"] - row["Low"]) / row["Last"] * 100)  
    prev_volume.append(row["Volume"])  
    prev_change_percent.append(row["%Chg"])  
    prev_vwap.append(None)  
    prev_trade_value.append(None)  
    percentage_change.append(None)  
# Create the DataFrame  
dataset = pd.DataFrame({    
    "Date": dates,    
    "day_of_the_week": day_of_the_week,    
    "Prev Open": prev_open,    
    "Prev Close": prev_close,    
    "Prev High": prev_high,    
    "Prev Low": prev_low,    
    "Prev HLPCT": prev_hlpct,    
    "Prev Volume": prev_volume,    
    "Prev Change Percent": prev_change_percent,    
    "Prev VWAP": prev_vwap,    
    "Prev Trade Value": prev_trade_value,    
    "percentage_change": percentage_change    
})  
  
dataset.head()  


# In[49]:


dataset.shape


# In[50]:


import pandas as pd

# Assuming you have a DataFrame named 'dataset' with your data
# Replace 'dataset' with your actual DataFrame name

# Convert the "Date" column to a datetime object
dataset['Date'] = pd.to_datetime(dataset['Date'])
print(dataset['Date'].max())


# In[51]:


print(dataset['Date'].min())


# #Manual calculation

# In[52]:


df


# In[53]:


df["date"]=pd.to_datetime(df["DateTime"]).dt.date


# In[54]:


f1=df[df["date"] == df["date"][0]]


# In[55]:


f1["Sentiment"].value_counts()


# In[56]:


num_of_positive = 367
num_of_negative = 578
num_of_neutral = 0


# In[57]:


print((num_of_positive - num_of_negative) / (num_of_positive + num_of_negative + num_of_neutral)) 


# In[58]:


# Sample DataFrame (stock_tweets)
import pandas as pd


stock_tweets = f1[["Timestamp", "Sentiment"]]

# Given timestamp and time intervals
open_timestamp = 1584964231000  # Example timestamp of interest
hour_1 = 0.5  # Adjusted time intervals for testing
hour_2 = 5.5  # Adjusted time intervals for testing

# Calculate sentiment index
sentiment_index = get_sentiment_index_three_class(stock_tweets, open_timestamp, hour_1, hour_2)

# Print the sentiment index
print(f"Sentiment Index: {sentiment_index}")


# In[59]:


get_ipython().run_cell_magic('time', '', 'def get_sentiment_index_three_class(concated, timestamp, hour_1, hour_2):  \n    filter1 = concated[\'Timestamp\'] < (timestamp - hour_1 * (60 * 60)) * 1000  \n    filter2 = concated["Timestamp"] > (timestamp - hour_2 * (60 * 60)) * 1000  \n    concated = concated.where(filter1 & filter2)  \n    tmp = concated.dropna(subset=["Timestamp"])  \n      \n    p = tmp.where(tmp[\'Sentiment\'] == \'positive\').dropna(subset=["Timestamp"])  \n    neg = tmp.where(tmp[\'Sentiment\'] == \'negative\').dropna(subset=["Timestamp"])  \n    neu = tmp.where(tmp[\'Sentiment\'] == \'neutral\').dropna(subset=["Timestamp"])  \n      \n    num_of_positive = p.count()["Sentiment"]  \n    num_of_negative = neg.count()["Sentiment"]  \n    num_of_neutral = neu.count()["Sentiment"]  \n  \n    return (num_of_positive - num_of_negative) / (num_of_positive + num_of_negative + num_of_neutral)  \n  \n# Preprocessing  \nfifteen_mins_df[\'day_of_the_week\'] = pd.to_datetime(fifteen_mins_df[\'Date\']).dt.dayofweek + 1  \n  \n# Initialize lists for features  \ndates = []  \nday_of_the_week = []  \npositive_percent_pre_market = []  \npositive_percent_55_135 = []  \npositive_percent_55_95 = []  \npositive_percent_95_135 = []  \npositive_percent_after_market = []  \npositive_percent_yesterday_market = []  \npositive_percent_the_day_before_yesterday = []  \nprev_open = []  \nprev_close = []  \nprev_high = []  \nprev_low = []  \nprev_hlpct = []  \nprev_volume = []  \nprev_change_percent = []  \nprev_vwap = []  \nprev_trade_value = []  \npercentage_change = []  \n  \n# Iterate through the rows of the fifteen_mins_df DataFrame  \nfor index, row in fifteen_mins_df.iterrows():  \n    # Convert the \'Time\' column to a Unix timestamp (in milliseconds)  \n    dt = pd.to_datetime(row["Date"].strftime(\'%Y-%m-%d\') + \' \' + row["Time"].strftime(\'%H:%M:%S\'))  \n    timestamp = int(dt.timestamp() * 1000)  \n  \n    # Calculate sentiment features using the second_df DataFrame and get_sentiment_index_three_class function  \n    sentiment_pre_market = get_sentiment_index_three_class(second_df, timestamp, 1, 4)  \n    sentiment_55_135 = get_sentiment_index_three_class(second_df, timestamp, 55, 135)  \n    sentiment_55_95 = get_sentiment_index_three_class(second_df, timestamp, 55, 95)  \n    sentiment_95_135 = get_sentiment_index_three_class(second_df, timestamp, 95, 135)  \n    sentiment_after_market = get_sentiment_index_three_class(second_df, timestamp, 2, 1)  \n    sentiment_yesterday_market = get_sentiment_index_three_class(second_df, timestamp, 3, 2)  \n    sentiment_day_before_yesterday = get_sentiment_index_three_class(second_df, timestamp, 4, 3)  \n  \n    # Append the values to the corresponding lists  \n    # ...  \n  \n  \n    # Append the values to the corresponding lists  \n    dates.append(row["Date"])  \n    day_of_the_week.append(row["day_of_the_week"])  \n    positive_percent_pre_market.append(sentiment_pre_market)  \n    positive_percent_55_135.append(sentiment_55_135)  \n    positive_percent_55_95.append(sentiment_55_95)  \n    positive_percent_95_135.append(sentiment_95_135)  \n    positive_percent_after_market.append(sentiment_after_market)  \n    positive_percent_yesterday_market.append(sentiment_yesterday_market)  \n    positive_percent_the_day_before_yesterday.append(sentiment_day_before_yesterday)  \n    prev_open.append(row["Open"])  \n    prev_close.append(row["Last"])\n    prev_high.append(row["High"])  \n    prev_low.append(row["Low"])  \n    prev_hlpct.append((row["High"] - row["Low"]) / row["Last"] * 100)  \n    prev_volume.append(row["Volume"])  \n    prev_change_percent.append(row["%Chg"])  \n    prev_vwap.append(None)  # Calculate the VWAP if required  \n    prev_trade_value.append(None)  # Calculate the trade value if required  \n    percentage_change.append(None)  # Calculate the percentage change if required  \n  \n# Create the DataFrame  \ndataset = pd.DataFrame({  \n    "Date": dates,  \n    "day_of_the_week": day_of_the_week,  \n    "positive_percent_pre_market": positive_percent_pre_market,  \n    "positive_percent_55_135": positive_percent_55_135,  \n    "positive_percent_55_95": positive_percent_55_95,  \n    "positive_percent_95_135": positive_percent_95_135,  \n    "positive_percent_after_market": positive_percent_after_market,  \n    "positive_percent_yesterday_market": positive_percent_yesterday_market,  \n    "positive_percent_the_day_before_yesterday": positive_percent_the_day_before_yesterday,  \n    "Prev Open": prev_open,  \n    "Prev Close": prev_close,  \n    "Prev High": prev_high,  \n    "Prev Low": prev_low,  \n    "Prev HLPCT": prev_hlpct,  \n    "Prev Volume": prev_volume,  \n    "Prev Change Percent": prev_change_percent,  \n    "Prev VWAP": prev_vwap,  \n    "Prev Trade Value": prev_trade_value,  \n    "percentage_change": percentage_change  \n})  \n  \ndataset.head()  \n\n    ')


# In[60]:


dataset.shape


# In[61]:


dataset.isnull().sum()


# In[ ]:





# In[62]:


dataset = df.copy()


# In[63]:


# Assuming you have imported pandas and defined your get_sentiment_index_three_class function

# Create an empty list to store the results
sentiment_indices = []

# Iterate through each row in the dataset DataFrame
for index, row in dataset.iterrows():
    open_timestamp = pd.to_datetime(row['DateTime']).timestamp() * 1000  # Convert to milliseconds
    
    # Convert hours to milliseconds
    hour_1 = 5.5 * 60 * 60 * 1000  # Replace with your desired values for the time intervals
    hour_2 = 9.5 * 60 * 60 * 1000  # Replace with your desired values for the time intervals
    
    sentiment_index = get_sentiment_index_three_class(concated, open_timestamp, hour_1, hour_2)
    
    # Append the result to the list
    sentiment_indices.append(sentiment_index)

# Add the sentiment indices to the dataset DataFrame
dataset['sentiment_index'] = sentiment_indices


# In[ ]:


dataset


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


processed_path = os.path.join('./drive/MyDrive/stocktwits_dataset', symbol, generated_file_name)
print(processed_path)

dataset = pd.read_csv(processed_path)
"""
dataset = dataset.drop(
    [
     'positive_percent_95_115',
     'positive_percent_115_135'
    ],
    axis=1
)"""
# Dele
dataset = dataset.drop(
    [
     #'sentiment_number_24hr',
     #'positive_percent_yesterday_market',
     #'Prev Open',
     #'Prev Close',
     #'Prev High',
     #'positive_percent_the_day_before_yesterday', #c
     #'Prev Low',
     #'Prev Volume',
     #'Prev Trade Value',
     'Prev VWAP',#c
     #'positive_percent_pre_market',
     'positive_percent_55_95',
     'positive_percent_55_135',
     'positive_percent_the_day_before_yesterday',
     'positive_percent_95_135',
     #'positive_percent_after_market',
     #'positive_percent_yesterday_market',
     'Prev HLPCT',#c
     #'Premarket Changed'
     #'positive_percent_55_135'#c,



     #"positive_percent_pre_market",
     #"positive_percent_after_market",
     #"positive_percent_yesterday_market"
    ],
    axis=1
)


dataset = dataset.drop(
    [
     'Unnamed: 0',
     #'day_of_the_week',
    ],
    axis=1
)
dataset["percentage_change"] = dataset["percentage_change"] * 100


days_of_week = pd.get_dummies(dataset.day_of_the_week)
dataset = pd.concat([dataset, days_of_week], axis=1)
temp_percentage_change = dataset["percentage_change"]
dataset = dataset.drop(["percentage_change"], axis=1)
dataset["percentage_change"] = temp_percentage_change
dataset = dataset.drop(['day_of_the_week'], axis=1)
dataset = dataset.drop(
    [
     1, 
     2, 
     3, 
     4, 
     5
    ],
    axis=1
)
end_index = dataset[dataset['Date']=='2022-02-28'].index[0]
start_testing_index = (len(dataset.index) - dataset[dataset['Date']=='2021-04-05'].index[0]) - 1
dataset = dataset[end_index:]
dataset = dataset.iloc[::-1]
print("End Index", end_index)
print("Start Testing Index", start_testing_index)
dataset


# # Traning

# In[ ]:


dataset.drop(["Date"], inplace=True, axis=1)
X, y = np.hsplit( dataset.to_numpy(), [-1])
_y = np.array(y>=0, dtype=int)


# ## Rolling Window

# In[ ]:


def movement_data(X, y, current, n):
  #X_train = X[:current]
  #y_train = y[:current]
  X_train = X[current-n:current]
  y_train = y[current-n:current]
  
  X_test = X[current]
  y_test = y[current]
  return (X_train, np.array([X_test]), y_train, np.array([y_test]))


# In[ ]:


from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import RobustScaler

history = []
history_w = []
history_score = []
range__ = [i for i in range(232, start_testing_index+1) if i % 2 == 0]

best_score = 0
best_n = 0
print(range__)
pred_his = []
c = 1
g = 'scale'
for i in range__:
  n=i
  score = 0
  count = 0
  preds = []
  while(True):
    try:
      start_num = n + count
      print(start_num, n)
      
      X_train, X_test, y_train, y_test = movement_data(X, _y, start_num, n)
      X_train, y_train = SMOTE(random_state=0, k_neighbors=5, n_jobs=-1).fit_resample(X_train, y_train)
      #標準化
      #scaler = StandardScaler()
      scaler = RobustScaler()
      scaler.fit(X_train) 
      X_train = scaler.transform(X_train)
      X_test = scaler.transform(X_test)

      #訓練      
      bagging_clf = BaggingClassifier(
          SVC(C=c, gamma=g),
          n_estimators=100,
          bootstrap=True,
          oob_score=False,
          n_jobs=-1,
          random_state=0
      ).fit(X_train, y_train)
      #計算分數 0 or 1
      pred = bagging_clf.predict(X_test)[0]
      preds.append(pred)
      real = y_test[0][0]

      count+=1
      #答對
      if pred == real:
        score+=1
      #print("Score:", score)
      history.append(
          score / count
      )
      print(score / count)
    except:
      print("some exception occur")
      break
  pred_his.append(preds)
  if (score/count) > best_score:
    best_score = score/count
    best_n = n
    best_c = c
    best_g = g
  history_w.append(n)
  history_score.append(score / count)
  print("N:", n)
  print("Current Accuracy:", score / count)
  print("Best N:", best_n)
  print("Best Score:", best_score)


# In[ ]:


pred_his =  np.array(pred_his)

f1_scores = []
for his in pred_his:
  labels = _y[start_testing_index:]
  preds = his[-(len(dataset)-start_testing_index):]
  #print( confusion_matrix(labels, preds) )
  #print( "Accuracy Score:", accuracy_score(labels, preds) )
  #print( "Precision。Score:", precision_score(labels, preds) )
  #print( "Recall Score:", recall_score(labels, preds) )
  print( "F1 Score:", f1_score(labels, preds) )
  f1_scores.append(f1_score(labels, preds))


# In[ ]:


sum(f1_scores) / len(f1_scores)


# In[ ]:


d = pd.DataFrame()
d["Window Size"] = [str(i) for i in range__]
d["F1 Score"] = f1_scores
d.plot(x="Window Size", y="F1 Score", kind="hist") 


# In[ ]:


d.plot(x="Window Size", y="F1 Score", kind="line") 


# ## Confusion Matrix

# In[ ]:


start = start_testing_index
_y = _y[start:]
preds = preds[- ( len(X) - start ):]
print( confusion_matrix(_y, preds) )
print( "Accuracy Score:", accuracy_score(_y, preds) )
print( "Precision Score:", precision_score(_y, preds) )
print( "Recall Score:", recall_score(_y, preds) )
print( "F1 Score:", f1_score(_y, preds) )


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




