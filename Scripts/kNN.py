import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics

def readCSV():
    dataSet = pd.read_csv("MultivariateInput.csv")
    return dataSet

def main():
    dataSet = readCSV()

    print(dataSet.describe())

    X = dataSet[['year', 'month', 'day', 'hour', 'minute', 'Celsius','PAH-1', 'PAH']].values
    y = dataSet['PAH+1'].values

    selFeat = ['year', 'month', 'day', 'hour', 'minute', 'Celsius','PAH-1', 'PAH']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    regressor = KNeighborsRegressor(n_neighbors=10)
    regressor.fit(X_train, y_train)

    #coefficient values for each feature in attributes
    #coeff_df = pd.DataFrame({'Coeff':regressor.coef_, "feature":selFeat})
    #print(coeff_df)

    y_pred = regressor.predict(X_test)

    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df1 = df.head(25)
    print(df1)

    df1.plot(kind='bar', figsize=(10, 8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()

    df2 = pd.DataFrame(X_test)
    df2.to_csv("pleasework.csv")
    df3 = pd.DataFrame(y_pred)
    df3.to_csv("pleasework2.csv")
    df4 = pd.DataFrame(y_test)
    df4.to_csv("pleasework3.csv")

    print('Confidence: ', regressor.score(X_test, y_test))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

if __name__ == "__main__":
    main()