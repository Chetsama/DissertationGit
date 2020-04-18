import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

def readCSV():
    dataSet = pd.read_csv("MultivariateInput.csv")
    return dataSet

def LinearModel(historicWindow, predictionWindow, dataSet):

    returnList = []

    print(dataSet.describe())

    X = dataSet[['year', 'month', 'day', 'hour', 'minute', 'Celsius', historicWindow]].values
    y = dataSet[predictionWindow].values

    selFeat = ['year', 'month', 'day', 'hour', 'minute', 'Celsius', historicWindow]

    # plt.figure(figsize=(15, 10))
    # plt.tight_layout()
    # seabornInstance.distplot(dataSet['PAH'])
    # plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    mlp = MLPRegressor(hidden_layer_sizes=(24, 24, 24), max_iter=1000)
    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)

    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df1 = df.head(25)
    print(df1)

    # print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    # print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    # print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
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
    PredictionList = ['PAH+1', 'PAH+2', 'PAH+3', 'PAH+4', 'PAH+6', 'PAH+12', 'PAH+24']
    for i in HistoricList:
        for j in PredictionList:

            output = LinearModel(i, j, dataSet)
            print(output)
            df.loc[len(df)] = output

    df.to_csv("MLPClassifierOutputResults242424-1000.csv", index=False, header=['HistoricWindow', 'PredictionWindow', 'MAE', 'MSE', 'RMSE'])
    print("done")

if __name__ == "__main__":
    main()