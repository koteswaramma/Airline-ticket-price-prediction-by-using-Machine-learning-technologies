import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os

import seaborn as sns


os.getcwd()

os.chdir(r'C:\Users\koti\Desktop\machine learning projects')


train = pd.read_excel('Data_Train.xlsx')

train.isna().sum()
train.columns


train.Route.value_counts()[0:10].plot.barh()

train.groupby(['Source','Destination',]).Price.count().plot.bar()


sns.boxplot(data=train,x='Source',y='Price')



train.shape
train.info()
train.head(5)

train_df = train[['Airline', 'Source', 'Destination', 'Total_Stops', 'Additional_Info', 'Date_of_Journey', 'Dep_Time', 
                  'Route', 'Arrival_Time', 'Price']]

Class = {'IndiGo': 'Economy',
         'GoAir': 'Economy',
         'Vistara': 'Economy',
         'Vistara Premium economy': 'Premium Economy',
         'Air Asia': 'Economy',
         'Trujet': 'Economy',
         'Jet Airways': 'Economy',
         'SpiceJet': 'Economy',
         'Jet Airways Business': 'Business',
         'Air India': 'Economy',
         'Multiple carriers': 'Economy',
         'Multiple carriers Premium economy': 'Premium Economy'}
train_df['Booking_Class'] = train_df['Airline'].map(Class)


market = {'IndiGo': 41.3,
         'GoAir': 8.4,
         'Vistara': 3.3,
         'Vistara Premium economy': 3.3,
         'Air Asia': 3.3,
         'Trujet': 0.1,
         'Jet Airways': 17.8,
         'SpiceJet': 13.3,
         'Jet Airways Business': 17.8,
         'Air India': 13.5,
         'Multiple carriers': 1,
         'Multiple carriers Premium economy': 1}
train_df['Market_Share'] = train_df['Airline'].map(market)


df1 = train_df.copy() 
df1['Day_of_Booking'] = '1/3/2019'
df1['Day_of_Booking'] = pd.to_datetime(df1['Day_of_Booking'],format='%d/%m/%Y')
df1['Date_of_Journey'] = pd.to_datetime(df1['Date_of_Journey'],format='%d/%m/%Y')
df1['Days_to_Departure'] = (df1['Date_of_Journey'] - df1['Day_of_Booking']).dt.days
train_df['Days_to_Departure'] = df1['Days_to_Departure']



del df1

train_df.head(2)

train_df['Arrival_Time'] = train['Arrival_Time'].str.split(' ').str[0]

def get_departure(dep):
    dep = dep.split(':')
    dep = int(dep[0])
    if (dep >= 6 and dep < 12):
        return 'Morning'
    elif (dep >= 12 and dep < 17):
        return 'Noon'
    elif (dep >= 17 and dep < 20):
        return 'Evening'
    else:
        return 'Night'
    
train_df['Dep_timeofday'] = train['Dep_Time'].apply(get_departure)   


train_df['Arr_timeofday'] = train['Arrival_Time'].apply(get_departure)    






train_df['Total_Stops'] = train_df['Total_Stops'].str.replace('non-stop','0')
train_df['Total_Stops'] = train_df['Total_Stops'].str.replace('stops','')
train_df['Total_Stops'] = train_df['Total_Stops'].str.replace('stop','')
train_df['Total_Stops'].fillna(0, inplace=True)   
train_df['Total_Stops'] = train_df['Total_Stops'].astype(float)



train_df['Hours'] = train['Duration'].str.split(' ').str[0]
train_df.drop(train_df[train_df['Hours'].str.contains('5m')].index,inplace=True)
train_df['Hours'] = train_df['Hours'].str.replace('h','').astype(float)
train_df['Hours'].fillna(0, inplace=True) 
train_df['Hours'].fillna(0, inplace=True) 

train_df['Minutes'] = train['Duration'].str.split(' ').str[1]
train_df['Minutes'] = train_df['Minutes'].str.replace('m','').astype(float)
train_df['Minutes'].fillna(0, inplace=True)



train_df['Hours'] = train_df['Hours'] * 60
train_df['Duration'] = train_df['Hours'] + train_df['Minutes']



train_df.drop(['Hours', 'Minutes'], axis=1, inplace=True)

train_df.head(2)



train_df['Price'] = np.log1p(train_df['Price'])

train_df['Duration'] = np.log1p(train_df['Duration'])



train_df['Additional_Info'] = train_df['Additional_Info'].str.replace('No info', 'No Info')



train_df = pd.get_dummies(train_df, columns=['Airline', 'Source', 'Destination', 'Additional_Info', 'Date_of_Journey',
                                             'Dep_Time', 'Arrival_Time', 'Dep_timeofday', 'Booking_Class', 'Arr_timeofday'],
                          drop_first=True)






def clean_route(route):
    route = str(route)
    route = route.split(' → ')
    return ' '.join(route)

train_df['Route'] = train_df['Route'].apply(clean_route)


from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(ngram_range=(1, 1), lowercase=False)
train_route = tf.fit_transform(train_df['Route'])

train_route = pd.DataFrame(data=train_route.toarray(), columns=tf.get_feature_names())




train_df = pd.concat([train_df, train_route], axis=1) 
train_df.drop('Route', axis=1, inplace=True)



train_df.head()


train_df.dropna(inplace=True)




train_df.shape




X = train_df.drop(labels=['Price'], axis=1)
y = train_df['Price'].values

from sklearn.model_selection import train_test_split
from math import sqrt 
from sklearn.metrics import mean_squared_log_error


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
from xgboost import XGBRegressor
xgb = XGBRegressor()
xgb.fit(X_train, y_train)

xgb.score(X_test,y_test)
xgb.score(X_train,y_train)
xgb.score(X,y)

'''
xgb.score(X_test,y_test)
Out[136]: 0.8721509775520547

xgb.score(X_train,y_train)
Out[137]: 0.8831597424027396

xgb.score(X,y)
Out[138]: 0.8803778738748554
'''


y_pred1 = xgb.predict(X_test)
print('RMSLE:', sqrt(mean_squared_log_error(np.exp(y_test), np.exp(y_pred1))))


'''
sqrt(mean_squared_log_error(np.exp(y_test), np.exp(y_pred1))))
RMSLE: 0.1846410412361866
'''




















df_sub = pd.DataFrame(data=y_pred1, columns=['Price'])
writer = pd.ExcelWriter('Output.xlsx', engine='xlsxwriter')
df_sub.to_excetest = pd.read_excel('Test_set.xlsx')


test_df = test[['Airline', 'Source', 'Destination', 'Total_Stops', 'Additional_Info', 'Date_of_Journey', 'Dep_Time', 
                'Route', 'Arrival_Time']]

test_df['Booking_Class'] = test_df['Airline'].map(Class)

test_df['Market_Share'] = test_df['Airline'].map(market)



df2 = test_df.copy() 
df2['Day_of_Booking'] = '1/3/2019'
df2['Day_of_Booking'] = pd.to_datetime(df2['Day_of_Booking'],format='%d/%m/%Y')
df2['Date_of_Journey'] = pd.to_datetime(df2['Date_of_Journey'],format='%d/%m/%Y')
df2['Days_to_Departure'] = (df2['Date_of_Journey'] - df2['Day_of_Booking']).dt.days
test_df['Days_to_Departure'] = df2['Days_to_Departure']


del df2


test_df['Arrival_Time'] = test['Arrival_Time'].str.split(' ').str[0]


test_df['Dep_timeofday'] = test['Dep_Time'].apply(get_departure)
test_df['Arr_timeofday'] = test['Arrival_Time'].apply(get_departure)  



test_df['Total_Stops'] = test_df['Total_Stops'].str.replace('non-stop','0')
test_df['Total_Stops'] = test_df['Total_Stops'].str.replace('stops','')
test_df['Total_Stops'] = test_df['Total_Stops'].str.replace('stop','')
#test_df['Total_Stops'].fillna(0, inplace=True)
test_df['Total_Stops'] = test_df['Total_Stops'].astype(float)



test_df['Hours'] = test['Duration'].str.split(' ').str[0]

test_df.drop(test_df[test_df['Hours'].str.contains('5m')].index,inplace=True)

test_df['Hours'] = test_df['Hours'].str.replace('h','').astype(float)
test_df['Hours'].fillna(0, inplace=True) 

test_df['Minutes'] = test['Duration'].str.split(' ').str[1]
test_df['Minutes'] = test_df['Minutes'].str.replace('m','').astype(float)
test_df['Minutes'].fillna(0, inplace=True)

test_df['Hours'] = test_df['Hours'] * 60
test_df['Duration'] = test_df['Hours'] + test_df['Minutes']

test_df.drop(['Hours', 'Minutes'], axis=1, inplace=True)
test_df['Duration'] = np.log1p(test_df['Duration'])


test_df['Additional_Info'] = test_df['Additional_Info'].str.replace('No info', 'No Info')




test_df = pd.get_dummies(test_df, columns=['Airline', 'Source', 'Destination', 'Additional_Info', 'Date_of_Journey',
                                           'Dep_Time', 'Arrival_Time', 'Dep_timeofday', 'Booking_Class', 'Arr_timeofday'],
                         drop_first=True)



test_df['Route'] = test_df['Route'].apply(clean_route)

test_route = tf.transform(test_df['Route'])

test_route = pd.DataFrame(data=test_route.toarray(), columns=tf.get_feature_names())




test_df = pd.concat([test_df, test_route], axis=1) 
test_df.drop('Route', axis=1, inplace=True)




test_df.head()





missing_cols_test = []
for col in X.columns:
    if col not in test_df.columns:
        missing_cols_test.append(col)
        
for i in missing_cols_test:
    test_df[i] = 0
    
    


test_df=test_df[X.columns]

np.unique(X.columns==test_df.columns)



Z_test = test_df

y_pred1 = xgb.predict(Z_test)



l(writer,sheet_name='Sheet1', index=False)
writer.save()


