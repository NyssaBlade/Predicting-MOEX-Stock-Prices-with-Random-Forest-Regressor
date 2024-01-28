import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import statistics
from datetime import datetime
import plotly.graph_objects as go
import warnings

def share_data_df(symbol, start_date, end_date):
    # Specify the security symbol and the desired market
    market = 'shares'

    # Initialize an empty list to store the rows
    all_rows = []

    # Set the initial start value
    start = 0

    while True:
        # Build the API URL with the start and limit parameters
        url = f'https://iss.moex.com/iss/history/engines/stock/markets/{market}/securities/{symbol}.json?' \
              f'from={start_date}&till={end_date}&start={start}&limit=100'

        # Send a GET request to the API
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()

            # Extract the relevant data from the response
            columns = data['history']['columns']
            rows = data['history']['data']

            # Append the rows to the list
            all_rows.extend(rows)

            # Increment the start value
            start += len(rows)

            # Check if there are more rows to fetch
            if len(rows) < 100:
                break
        else:
            print('Failed to fetch data from the MOEX API')
            return None

    # Create a DataFrame from all the fetched rows
    df = pd.DataFrame(all_rows, columns=columns)

    df.drop_duplicates(subset='TRADEDATE', keep='last', inplace=True)
    df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
    df['YEAR'] = df['TRADEDATE'].dt.year
    df[['OPEN LAGGED', 'LOW LAGGED', 'HIGH LAGGED', 'CLOSE LAGGED', 'WAPRICE LAGGED']] = df[
        ['OPEN', 'LOW', 'HIGH', 'CLOSE', 'WAPRICE']].shift(+1)
    df = df[['TRADEDATE', 'OPEN', 'LOW', 'HIGH', 'CLOSE', 'OPEN LAGGED', 'LOW LAGGED', 'HIGH LAGGED', 'CLOSE LAGGED']].fillna(0)

    # Calculate the correlation matrix
    corr_matrix = df.corr()

    # Generate the correlation heatmap
    plt.figure(figsize=(10, 5))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Between SNGS Market Indicators')
    plt.show()

    print(df.head())
    return df

symbol = 'POLY'  
start_date = '2016-01-01'
end_date = '2024-01-30'

df=share_data_df(symbol, start_date, end_date)

parameters_high=parameters(df,['OPEN', 'CLOSE LAGGED', 'HIGH LAGGED'],'HIGH')
parameters_low=parameters(df,['OPEN', 'CLOSE LAGGED', 'LOW LAGGED'],'LOW')

best_max_depth_h=parameters_high[0]
best_n_estimators_h=parameters_high[1]
best_random_state_h=parameters_high[2]

best_max_depth_l=parameters_low[0]
best_n_estimators_l=parameters_low[1]
best_random_state_l=parameters_low[2]

def high(open, df):
    X = df[['OPEN', 'CLOSE LAGGED', 'HIGH LAGGED']]
    X, y = X, df['HIGH']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=2)

    regressor = RandomForestRegressor(max_depth=best_max_depth_h, n_estimators=best_n_estimators_h, random_state=best_random_state_h)

    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)

    # Calculate additional performance metrics
    r2 = r2_score(y_test, y_pred)

    # Print the additional performance metrics
    print("\n")
    print(f"\033[7mR^2 Score:\033[0m \033[38;5;196m{round(r2,3)}\033[0m")

    index = df.tail(1).index.tolist()
    lagged_indicators = df[['CLOSE','HIGH']].loc[index].values.tolist()[0]

    to_predict = np.array([[open] + lagged_indicators])

    diff=y_pred - y_test
    min_diff=min(diff)
    max_diff=max(diff)
    mean_diff=statistics.mean(diff)   
    median_diff=statistics.median(diff)      



    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prediction = regressor.predict(to_predict)

        print(f"\033[47mModel Predicted Daily Highest Price:\033[0m from {round(prediction[0]-median_diff,2)} to {round(prediction[0]+median_diff,2)}")

def low(open, df):
    X = df[['OPEN', 'CLOSE LAGGED','LOW LAGGED']]
    X, y = X, df['LOW']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=2)

    regressor = RandomForestRegressor(max_depth=best_max_depth_l, n_estimators=best_n_estimators_l, random_state=best_random_state_l)

    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)

    # Calculate additional performance metrics
    r2 = r2_score(y_test, y_pred)

    # Print the additional performance metrics
    print("\n")
    print(f"\033[7mR^2 Score:\033[0m \033[38;5;196m{round(r2,3)}\033[0m")

    index = df.tail(1).index.tolist()
    lagged_indicators = df[['CLOSE', 'LOW']].loc[index].values.tolist()[0]

    to_predict = np.array([[open] + lagged_indicators])

    diff= y_pred - y_test
    min_diff=min(diff)
    max_diff=max(diff)
    mean_diff=statistics.mean(diff)   
    median_diff=statistics.median(diff)      



    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prediction = regressor.predict(to_predict)

        print(f"\033[47mModel Predicted Daily Lowes Price:\033[0m from {round(prediction[0]+median_diff,2)} to {round(prediction[0]-median_diff,2)}")

def price(open,df):
    high(open, df)
    low(open, df) 
