import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn import utils

def main():
    dataTrain = pd.read_csv("harbourPAHtrain.csv")
    dataTest = pd.read_csv("harbourPAHtest.csv")
    # print df.head()

    x_train = dataTrain[['year', 'month', 'day', 'hour', 'minute']].to_numpy().reshape(-1,5)
    y_train = dataTrain['PAH']
    y_train = y_train.astype('int')

    x_test = dataTest[['year', 'month', 'day', 'hour', 'minute']].to_numpy().reshape(-1,5)
    y_test = dataTest['PAH']

    knn = KNeighborsClassifier()
    model = knn.fit(x_train, y_train)

    output = {'PredictedPAH': model.predict(x_test)}
    df = pd.DataFrame(output, columns=['PredictedPAH'])

    df.to_csv(r'harbourPAHoutput.csv', index=True, header=True)
    print(df)

if __name__ == "__main__":
    main()