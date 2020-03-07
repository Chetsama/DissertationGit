import pandas as pd
import numpy as np

def importCSV():
    df = pd.read_csv("dataCleanTest.csv", names = ['year', 'month', 'day', 'hour', 'minute', 'PAH'])
    return df

def outputCSV(df):

    for index, rows in df.iterrows():
        if (pd.isnull(rows['PAH'])):
            print(df.loc[index-1, ['PAH']])


def main():
    df = importCSV()
    outputCSV(df)

if __name__ == "__main__":
    main()