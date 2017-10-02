
import math

class NaiveBayes:
    def __init__(self):
        pass

    def mean(self, data):
        return sum(data)/float(data.__len__())
        pass

    def stddev(self, data):
        average = self.mean(data)
        variance = sum( pow(d - average , 2) for d in data ) / float(data.__len__() - 1)
        return math.sqrt(variance)
        pass


    #Here input dataset is rows of list of attributes corresponding to a particular class.
    #We need to find the mean and standard deviation for all the attributes. and return it
    def findMeanStddev(self, dataset):
        mean_stddev = [ (self.mean(attr), self.stddev(attr)) for attr in zip(*dataset)]
        del mean_stddev[-1] #to remove mean and stddev of class labels
        return mean_stddev
        pass


    def splitDataByClass(self, datasetToBeSplitByClass):
        dataByClass = {}
        for i in range(datasetToBeSplitByClass.__len__()):
            rowData = datasetToBeSplitByClass[i]
            classLabel = rowData[-1]
            if classLabel not in dataByClass:
                dataByClass[classLabel] = []
            dataByClass[classLabel].append(rowData)

        return dataByClass



    def createModel(self, dataset):

        #Get dictionary with class labels as keys and other attributes list as values using splitDataByClass
        classDict = self.splitDataByClass(dataset)
        model = {} # used to store mean and standard deviation for each classLabel
        for classLabel, values in classDict.iteritems():
            #values has data corresponding to classLabel. This data is row of list of attributes
            model[classLabel] = self.findMeanStddev(values)

        #model has tuple of mean and standard deviation for all the attributes and all class labels.
        return model



    def calculateProbability(self, data, mean, stddev):
        exp = math.exp(-(math.pow(data - mean , 2) / (2 * math.pow(stddev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stddev)) * exp



    #For each test data calculateClassProbability.
    #data here is 1 row of test data with all the attributes
    #model here is list of tuples of mean and std dev. One tuple in the list corresponds to each attributes.
    #This function assumes conditional independence among the attributes to calculate the probability using Naive Bayes.
    def calculateClassProbabilities(self, model, data): # list of tuples of mean and stddev
        classProbabilities = {}
        for classLabel, values in model.iteritems():
            classProbabilities[classLabel] = 1
            for index in range(values.__len__()): # run till number of attributes
                mean, stddev = values[index]
                dataX = data[index]
                classProbabilities[classLabel] = classProbabilities[classLabel] * self.calculateProbability(dataX, mean, stddev)
        return classProbabilities



    def predict(self, model, data):
        classProbabilities = self.calculateClassProbabilities(model, data)
        predictedLabel = None
        predProbability = -1
        for classLabel, prob in classProbabilities.iteritems():
            if predictedLabel is None or prob > predProbability:
                predictedLabel = classLabel
                predProbability = prob
        return predictedLabel




    def getPredictions(self, model, testData):
        predictedLabels = []
        for index in range(len(testData)):
            result = self.predict(model, testData[index])
            predictedLabels.append(result)
        return predictedLabels



    def getAccuracy(self, testData, predictions):
        correctVal = 0
        for index in range(len(testData)):
            if testData[index][-1] == predictions[index]:
                correctVal += 1
        return (correctVal / float(len(testData))) * 100