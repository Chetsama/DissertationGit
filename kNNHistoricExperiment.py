import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics

def readCSV():
    dataSet = pd.read_csv("LinearMultivariateInput.csv")
    return dataSet

def LinearModel(historicWindow, predictionWindow, dataSet):

    returnList = []

    print(dataSet.describe())

    X = dataSet[['year', 'month', 'day', 'hour', 'minute', 'PAH', historicWindow]].values
    y = dataSet[predictionWindow].values

    selFeat = ['year', 'month', 'day', 'hour', 'minute', 'PAH', historicWindow]

    # plt.figure(figsize=(15, 10))
    # plt.tight_layout()
    # seabornInstance.distplot(dataSet['PAH'])
    # plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    regressor = KNeighborsRegressor(n_neighbors=10)
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)

    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df1 = df.head(25)
    print(df1)

    df1.plot(kind='bar', figsize=(10, 8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    #plt.show()

    # print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    # print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    # print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    #returnList.append(movingAverage)
    returnList.append(historicWindow)
    returnList.append(predictionWindow)
    returnList.append(metrics.mean_absolute_error(y_test, y_pred))
    returnList.append(metrics.mean_squared_error(y_test, y_pred))
    returnList.append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    return returnList

def main():
    dataSet = readCSV()
    df = pd.DataFrame(columns=['HistoricWindow', 'PredictionWindow', 'MAE', 'MSE', 'RMSE'])
    HistoricList = ['PAH-24', 'PAH-12', 'PAH-6', 'PAH-3', 'PAH-1']
    PredictionList = ['quarter', 'half', 'PAH+1', 'PAH+2', 'PAH+3', 'PAH+4', 'PAH+6', 'PAH+12', 'PAH+24']
    movingAverage = ['MPAH-6', 'MPAH-3']

    for i in HistoricList:
        for j in PredictionList:
            output = LinearModel(i, j, dataSet)
            print(output)
            df.loc[len(df)] = output

    df.to_csv("LinearImputationkNNOutput.csv", index=False, header=['HistoricWindow', 'PredictionWindow', 'MAE', 'MSE', 'RMSE'])
    print("done")

if __name__ == "__main__":
    main()