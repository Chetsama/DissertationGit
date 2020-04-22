import pandas as pd
import numpy as np


def importCSV():
    df = pd.read_csv("LinearCleanerTest.csv", names=['year', 'month', 'day', 'hour', 'minute', 'PAH', 'PAH-24', 'PAH-12', 'PAH-6', 'PAH-3', 'PAH-1', 'PAH+1', 'PAH+2', 'PAH+3', 'PAH+4', 'PAH+6', 'PAH+12', 'PAH+24', 'half', 'quarter'])
    return df


def outputCSV(df):
    # last = 0
    # curIndex = 0
    # next = 0
    # lastIndex = 0
    startNewList = True

    historyList = ['PAH', 'PAH-24', 'PAH-12', 'PAH-6', 'PAH-3', 'PAH-1', 'PAH+1', 'PAH+2', 'PAH+3', 'PAH+4', 'PAH+6', 'PAH+12', 'PAH+24', 'half', 'quarter']

    nullList = []
    tempList = []

    for k in historyList:
        for index, rows in df.iterrows():
            stop = False
            counter = 0
            if (pd.isnull(rows[k])):
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
                tempList = []
        print(nullList)

        for i in nullList:
            first = df.loc[i[0]-1, [k]]
            last = df.loc[i[len(i)-1]+1, [k]]
            #todo write function that replaces with interesting values
            #average = (float(first)+float(last)) /2
            print("len " + str(len(i)))
            m = ((float(last)-float(first))/((len(i)+1)))
            print(m)
            counter = 1
            for j in i:
                print(m*counter+float(first))
                df.loc[j,[k]] = m*counter+float(first)
                print(df.loc[j,[k]])
                counter = counter + 1


        nullList = []
        #link to multi-variate regression
        df.to_csv("test.csv", index=False, header=k)

def main():
    df = importCSV()
    outputCSV(df)


if __name__ == "__main__":
    main()
