import pandas as pd

class ParseData:
    def __init__(self, trainSetPath, testSetPath):
        self._trainSetPath = trainSetPath
        self._testSetPath = testSetPath
        self.fullTrainSet = None
        self.fullTestSet = None
        self.trainSetBusiness = None
        self.trainSetFilm = None
        self.trainSetFootball = None
        self.trainSetPolitics = None
        self.trainSetTechnology = None

    #Parse data from CSV files
    def parse(self):
        print("Hello trainSetPath is : " + self._trainSetPath + " testSetPath is : " + self._testSetPath)
        self.fullTrainSet = pd.read_csv(self._trainSetPath, sep='\t')
        self.fullTestSet = pd.read_csv(self._testSetPath, sep='\t')

    def categorizeData(self):
        self.trainSetBusiness = self.fullTrainSet[self.fullTrainSet["Category"] == "Business"]
        self.trainSetFilm = self.fullTrainSet[self.fullTrainSet["Category"] == "Film"]
        self.trainSetFootball = self.fullTrainSet[self.fullTrainSet["Category"] == "Football"]
        self.trainSetPolitics = self.fullTrainSet[self.fullTrainSet["Category"] == "Politics"]
        self.trainSetTechnology = self.fullTrainSet[self.fullTrainSet["Category"] == "Technology"]