import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def readCSV():
    dataSet = pd.read_csv("TemperatureVsPAH.csv")
    return dataSet

def main():
    dataSet = readCSV()
    print(dataSet.describe())

    dataSet.plot(x='Celsius', y='PAH', style='o')
    plt.title('Celsius vs PAH')
    plt.xlabel('Celsius')
    plt.ylabel('PAH')
    #plt.show()

    plt.figure(figsize=(15,10))
    plt.tight_layout()
    seabornInstance.distplot(dataSet['PAH'])
    #plt.show()

    X = dataSet['Celsius'].values.reshape(-1,1)
    y = dataSet['PAH'].values.reshape(-1,1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    regressor = LinearRegression()
    regressor.fit(X_train,y_train)

    print(regressor.intercept_)
    print(regressor.coef_)

    y_pred = regressor.predict(X_test)
    df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted':y_pred.flatten()})

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

if __name__ == "__main__":
    main()