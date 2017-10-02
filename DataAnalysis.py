import dataset as ds
import NaiveBayes #user defined module


if __name__ == "__main__":

    dataset = ds.Dataset()
    nb = NaiveBayes.NaiveBayes()

    data = dataset.loadDataFile("pima-indians-diabetes.csv")

    training_Data, test_Data = dataset.splitDataset(data, 0.70)

    print('Dataset has {0} rows. Split into training data with {1} rows and test data with {2} rows').format(data.__len__(), training_Data.__len__(), test_Data.__len__())


    model = nb.createModel(training_Data)
    predictedLabels = nb.getPredictions(model, test_Data)
    accuracy = nb.getAccuracy(test_Data, predictedLabels)

    print('Accuracy: {0}%').format(accuracy)






