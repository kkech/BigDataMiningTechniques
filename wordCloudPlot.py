import pandas as pd
from parseData import ParseData
import os as os
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS


class WordCloudPlot:
    def __init__(self, parseData, businessPlotSavePath, filmPlotSavePath, footballPlotSavePath, politicsPlotSavePath, technologyPlotSavePath, allPlotSavePath):
        self.parseData = parseData
        self.businessPlotSavePath = businessPlotSavePath
        self.filmPlotSavePath = filmPlotSavePath
        self.footballPlotSavePath = footballPlotSavePath
        self.politicsPlotSavePath = politicsPlotSavePath
        self.technologyPlotSavePath = technologyPlotSavePath
        self.allPlotSavePath = allPlotSavePath
        STOPWORDS.add("said")
        STOPWORDS.add("will")
        STOPWORDS.add("say")

    #Plot Word Cloud for Business Category
    def plotWordCloudBusiness(self):
        trainSetBusinessContentColumn = self.parseData.trainSetBusiness[["Content"]]
        A = np.array(trainSetBusinessContentColumn)
        text = ""
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                text += str(A[i, j]) + ","

        wordcloud = WordCloud(stopwords=STOPWORDS,
                              background_color='white',
                              width=1800,
                              height=1400
                              ).generate(text)

        # Display the generated image:
        # the matplotlib way:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        directory = os.path.dirname(self.businessPlotSavePath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(self.businessPlotSavePath)
        # plt.show()

    #Plot Word Cloud for Film Category
    def plotWordCloudFilm(self):
        trainSetFilmContentColumn = self.parseData.trainSetFilm[["Content"]]
        A = np.array(trainSetFilmContentColumn)
        text = ""
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                text += str(A[i, j]) + ","

        wordcloud = WordCloud(stopwords=STOPWORDS,
                              background_color='white',
                              width=1800,
                              height=1400).generate(text)

        # Display the generated image:
        # the matplotlib way:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        directory = os.path.dirname(self.filmPlotSavePath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(self.filmPlotSavePath)
        # plt.show()

    #Plot Word Cloud for Football Category
    def plotWordCloudFootball(self):
        trainSetFootballContentColumn = self.parseData.trainSetFootball[["Content"]]
        A = np.array(trainSetFootballContentColumn)
        text = ""
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                text += str(A[i, j]) + ","

        wordcloud = WordCloud(stopwords=STOPWORDS,
                              background_color='white',
                              width=1800,
                              height=1400).generate(text)

        # Display the generated image:
        # the matplotlib way:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        directory = os.path.dirname(self.footballPlotSavePath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(self.footballPlotSavePath)
        # plt.show()

    # Plot Word Cloud for Politics Category
    def plotWordCloudPolitics(self):
        trainSetPoliticsContentColumn = self.parseData.trainSetPolitics[["Content"]]
        A = np.array(trainSetPoliticsContentColumn)
        text = ""
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                text += str(A[i, j]) + ","

        wordcloud = WordCloud(stopwords=STOPWORDS,
                              background_color='white',
                              width=1800,
                              height=1400).generate(text)

        # Display the generated image:
        # the matplotlib way:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        directory = os.path.dirname(self.politicsPlotSavePath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(self.politicsPlotSavePath)
        # plt.show()

    # Plot Word Cloud for Technology Category
    def plotWordCloudTechnology(self):
        trainSetTechnologyContentColumn = self.parseData.trainSetTechnology[["Content"]]
        A = np.array(trainSetTechnologyContentColumn)
        text = ""
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                text += str(A[i, j]) + ","

        wordcloud = WordCloud(stopwords=STOPWORDS,
                              background_color='white',
                              width=1800,
                              height=1400).generate(text)

        # Display the generated image:
        # the matplotlib way:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        directory = os.path.dirname(self.technologyPlotSavePath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(self.technologyPlotSavePath)
        # plt.show()

    def plotWordCloudAll(self):
        trainSetBusinessContentColumn = self.parseData.trainSetBusiness[["Content"]]
        A = np.array(trainSetBusinessContentColumn)
        text = ""
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                text += str(A[i, j]) + ","

        trainSetFilmContentColumn = self.parseData.trainSetFilm[["Content"]]
        A = np.array(trainSetFilmContentColumn)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                text += str(A[i, j]) + ","

        trainSetFootballContentColumn = self.parseData.trainSetFootball[["Content"]]
        A = np.array(trainSetFootballContentColumn)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                text += str(A[i, j]) + ","

        trainSetPoliticsContentColumn = self.parseData.trainSetPolitics[["Content"]]
        A = np.array(trainSetPoliticsContentColumn)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                text += str(A[i, j]) + ","

        trainSetTechnologyContentColumn = self.parseData.trainSetTechnology[["Content"]]
        A = np.array(trainSetTechnologyContentColumn)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                text += str(A[i, j]) + ","

        wordcloud = WordCloud(stopwords=STOPWORDS,
                              background_color='white',
                              width=1800,
                              height=1400).generate(text)

        # Display the generated image:
        # the matplotlib way:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        directory = os.path.dirname(self.allPlotSavePath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(self.allPlotSavePath)
