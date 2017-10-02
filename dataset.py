
import math

class Dataset:
    def __init__(self):
        self.loadedDataSet = None
        self.train_Data = None
        self.test_Data = None


    def loadDataFile(self, filename):
        import csv
        dataFileObject = open(filename, "rb")
        data = csv.reader(dataFileObject)
        rows = list(data)
        #print(rows)
        #print("\n\n")
        dataFileObject.close()

        for rowIndex in range(rows.__len__()):
            rows[rowIndex] = [float(column) for column in rows[rowIndex]]

        self.loadedDataSet = rows

        return self.loadedDataSet




    def splitDataset(self, datasetToBeSplit, splitRatio):
        train_Len = int(datasetToBeSplit.__len__() * splitRatio)
        test_Len = datasetToBeSplit.__len__() - train_Len
        
        self.train_Data = datasetToBeSplit[:train_Len]
        self.test_Data = datasetToBeSplit[train_Len:]

        return  [self.train_Data, self.test_Data]


