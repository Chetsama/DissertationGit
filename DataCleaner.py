import pandas as pd
import numpy as np


def importCSV():
    df = pd.read_csv("dataCleanTest.csv", names=['year', 'month', 'day', 'hour', 'minute', 'PAH'])
    return df


def outputCSV(df):
    # last = 0
    # curIndex = 0
    # next = 0
    # lastIndex = 0
    startNewList = True
    nullList = []
    tempList = []

    for index, rows in df.iterrows():
        stop = False
        counter = 0
        if (pd.isnull(rows['PAH'])):
            if startNewList:
                startNewList = False
                tempList = []
                tempList.append(index)
            else:
                tempList.append(index)
        else:
            if(startNewList==False):
                if len(tempList)!=0:
                    nullList.append(tempList)
            startNewList = True
    print(nullList)

    for i in nullList:
        first = df.loc[i[0]-1, ['PAH']]
        last = df.loc[i[len(i)-1]+1, ['PAH']]
        #todo write function that replaces with interesting values
        average = (float(first)+float(last)) /2
        for j in i:
            df.loc[j,['PAH']] = average
            print(df.loc[j,['PAH']])

    df.to_csv("dataCleanTestOut.csv", index=False, header=False)

def main():
    df = importCSV()
    outputCSV(df)


if __name__ == "__main__":
    main()
