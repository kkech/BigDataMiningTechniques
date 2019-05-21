import multiprocessing
import string

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk.stem import LancasterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn import linear_model
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from duplicates import Duplicates
from parseData import ParseData
from wordCloudPlot import WordCloudPlot


class Tokenizer(object):
    def __init__(self):
        self.tok = RegexpTokenizer(r'some_regular_expression')
        self.stemmer = LancasterStemmer()
    def __call__(self, doc):
        return [self.stemmer.stem(token)
                for token in self.tok.tokenize(doc)]

def clean(s):
    return [w.strip(',."!?:;()\'') for w in s]


def w2v(data):
    print("------------------  W2V  ------------------\n")

    sentences = [row['Content'].split() for _, row in data.iterrows()]

    doc_word = []
    for i in range(0, data['Content'].count()):
        doc_word.append(str(data['Content'][i]).translate(string.punctuation).split())

    model = Word2Vec(doc_word, workers=8, size=200, min_count=10)
    w2v = dict(zip(model.wv.index2word, model.wv.syn0))

    return np.array([
        np.mean([w2v[w] for w in words if w in w2v]
                or [np.zeros(200)], axis=0) for words in doc_word
    ])

class Run:

    #####Gen Params
    trainSetPath = "/Users/kechagiaskonstantinos/Downloads/Datasets-2018/train_set.csv"
    testSetPath = "/Users/kechagiaskonstantinos/Downloads/Datasets-2018/test_set.csv"

    businessPlotSavePath = "/Users/kechagiaskonstantinos/Downloads/Datasets-2018/pic/businessPng.png"
    filmPlotSavePath = "/Users/kechagiaskonstantinos/Downloads/Datasets-2018/pic/filmPng.png"
    footballPlotSavePath = "/Users/kechagiaskonstantinos/Downloads/Datasets-2018/pic/footballPng.png"
    politicsPlotSavePath = "/Users/kechagiaskonstantinos/Downloads/Datasets-2018/pic/politicsPng.png"
    technologyPlotSavePath = "/Users/kechagiaskonstantinos/Downloads/Datasets-2018/pic/technologyPng.png"
    allPlotSavePath = "/Users/kechagiaskonstantinos/Downloads/Datasets-2018/pic/allPng.png"
    #####

    parseData = ParseData(trainSetPath,testSetPath)
    parseData.parse()
    parseData.categorizeData()
    # print(parseData.fullTrainSet)
    # print(parseData.fullTestSet)
    #
    # print(parseData.trainSetBusiness)
    # print(parseData.trainSetFilm)
    # print(parseData.trainSetFootball)
    # print(parseData.trainSetPolitics)
    # print(parseData.trainSetTechnology)

    for column in parseData.fullTrainSet.columns:
        print(column)

    # Make Wordclouds

    print("Starting Making WordClouds")

    trainSetBusinessContentColumn = parseData.trainSetBusiness[["Content"]]

    wordcloudplot = WordCloudPlot(parseData, businessPlotSavePath, filmPlotSavePath, footballPlotSavePath,
                                  politicsPlotSavePath, technologyPlotSavePath, allPlotSavePath)
    wordcloudplot.plotWordCloudBusiness()
    wordcloudplot.plotWordCloudFilm()
    wordcloudplot.plotWordCloudFootball()
    wordcloudplot.plotWordCloudPolitics()
    wordcloudplot.plotWordCloudTechnology()
    wordcloudplot.plotWordCloudAll()

    print("Making WordClouds Completed")

    # Find Duplicate

    print("Start Finding Duplicates")

    duplicates = Duplicates.findDuplicates(parseData.fullTrainSet)
    the_list = []

    for i in range(0, len(duplicates) - 1):
        # print(duplicates[i])
        for j in range(i, len(duplicates[i]) - 1):
            if duplicates[i][j] > 0.7:
                # print(duplicates[i][j])
                # print(parseData.fullTrainSet.iloc[i]['Id'])
                if parseData.fullTrainSet.iloc[i]['Id'] != parseData.fullTrainSet.iloc[j]['Id']:
                    the_list.append([str(parseData.fullTrainSet['Id'][i]), str(parseData.fullTrainSet['Id'][j]),
                                     str(duplicates[i][j])])

    df = pd.DataFrame(the_list, columns=['Document1', 'Document2', "Similarity"])
    df.to_csv("/Users/kechagiaskonstantinos/Downloads/Datasets-2018/duplicates.csv", sep=',', encoding='utf-8',
              index=False)

    print("Finding Duplicates Completed")


    #Check SVD variance
    # start = time()
    # svd = TruncatedSVD(n_components=3000,
    #                          algorithm='randomized',
    #                          n_iter=10, random_state=42)
    #
    # textClf = Pipeline([
    #     ('c', CountVectorizer()),
    #     ('svm', svd)
    # ])
    # textClf.fit(parseData.fullTrainSet['Content'], parseData.fullTrainSet['Category'])
    # end = time()
    #
    # print(end - start)
    # print(svd.explained_variance_ratio_.sum())

    # Find using myModel
    print("Start prediction")
    the_list = []

    textLRClf = Pipeline([
        ('vect', CountVectorizer(min_df=1, stop_words='english')),
        ('tfidf', TfidfTransformer()),
        ('clf', linear_model.SGDClassifier(max_iter=1000, tol=1e-3))

    ])
    textLRClf.fit(parseData.fullTrainSet['Content'], parseData.fullTrainSet['Category'])
    predicted = textLRClf.predict(parseData.fullTestSet['Content'])
    for i in range(0, len(predicted) - 1):
        the_list.append([str(parseData.fullTestSet['Id'][i]), predicted[i]])
        print(str(parseData.fullTestSet['Id'][i]) + " predicted category : " + predicted[i])

    df = pd.DataFrame(the_list, columns=['Test_Document_ID', 'Predicted_Category'])
    df.to_csv("/Users/kechagiaskonstantinos/Downloads/Datasets-2018/testSet_categories.csv", sep=',',
              encoding='utf-8',
              index=False)

    print("Prediction completed")


    scoring = ["accuracy", 'precision_macro', 'recall_macro']


    X_train, X_test, y_train, y_test = train_test_split(parseData.fullTrainSet['Content'], parseData.fullTrainSet['Category'], test_size=0.4, random_state=0)

    the_list = [['Accuracy'],['Precision'],['Recall']]


    #Evaluate Bow and RandomForestClassifier
    textRfc = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('rfc', RandomForestClassifier(n_estimators=100)),
    ])

    print("###Bow and RandomForestClassifier###")
    scores = cross_validate(textRfc, parseData.fullTrainSet['Content'], parseData.fullTrainSet['Category'], scoring=scoring,
                            cv=10, return_train_score=False)

    the_list[0].append((scores['test_accuracy'].mean()))
    the_list[1].append((scores['test_precision_macro'].mean()))
    the_list[2].append((scores['test_recall_macro'].mean()))


    print("Accuracy: %0.4f (+/- %0.2f)" % (scores['test_accuracy'].mean(), scores['test_accuracy'].std() * 2))
    print("Recall: %0.4f (+/- %0.2f)" % (scores['test_recall_macro'].mean(), scores['test_recall_macro'].std() * 2))
    print("Precision: %0.4f (+/- %0.2f)" % (scores['test_precision_macro'].mean(), scores['test_precision_macro'].std() * 2))
    print("######")


    #Evaluate Bow and SVM
    print("###Bow and SVM###")

    textClf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('svm', LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
                  intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                  multi_class='ovr', penalty='l2', random_state=0, tol=1e-05, verbose=0))
    ])

    textClf.fit(X_train, y_train)
    print(textClf.score(X_test, y_test))

    scores = cross_validate(textClf, parseData.fullTrainSet['Content'], parseData.fullTrainSet['Category'],
                            scoring=scoring,
                            cv=10, return_train_score=False)

    the_list[0].append((scores['test_accuracy'].mean()))
    the_list[1].append((scores['test_precision_macro'].mean()))
    the_list[2].append((scores['test_recall_macro'].mean()))

    print("Accuracy: %0.4f (+/- %0.2f)" % (scores['test_accuracy'].mean(), scores['test_accuracy'].std() * 2))
    print("Recall: %0.4f (+/- %0.2f)" % (scores['test_recall_macro'].mean(), scores['test_recall_macro'].std() * 2))
    print("Precision: %0.4f (+/- %0.2f)" % (scores['test_precision_macro'].mean(), scores['test_precision_macro'].std() * 2))
    print("######")

    # Svd - SVM

    textSvdClf = Pipeline([
        ('vect', CountVectorizer(min_df=1, stop_words='english')),
        ('svd', TruncatedSVD(n_components=3000,
                             algorithm='randomized',
                             n_iter=10, random_state=42)),
        ('clf', LinearSVC(random_state=0, tol=1e-5))
    ])
    print("###Svd and SVM###")

    scores = cross_validate(textSvdClf, parseData.fullTrainSet['Content'], parseData.fullTrainSet['Category'],
                            scoring=scoring,
                            cv=10, return_train_score=False)

    the_list[0].append((scores['test_accuracy'].mean()))
    the_list[1].append((scores['test_precision_macro'].mean()))
    the_list[2].append((scores['test_recall_macro'].mean()))

    print("Accuracy: %0.4f (+/- %0.2f)" % (scores['test_accuracy'].mean(), scores['test_accuracy'].std() * 2))
    print("Recall: %0.4f (+/- %0.2f)" % (scores['test_recall_macro'].mean(), scores['test_recall_macro'].std() * 2))
    print("Precision: %0.4f (+/- %0.2f)" % (scores['test_precision_macro'].mean(), scores['test_precision_macro'].std() * 2))
    print("######")

    #SVD - RFC

    print("###Svd and RFC###")

    textSvdRfc = Pipeline([
        ('vect', CountVectorizer(min_df=1, stop_words='english')),
        ('svd', TruncatedSVD(n_components=3000,
                             algorithm='randomized',
                             n_iter=10, random_state=42)),
        ('clf', RandomForestClassifier(n_estimators=100))
    ])

    scores = cross_validate(textSvdRfc, parseData.fullTrainSet['Content'], parseData.fullTrainSet['Category'],
                            scoring=scoring,
                            cv=10, return_train_score=False)

    the_list[0].append((scores['test_accuracy'].mean()))
    the_list[1].append((scores['test_precision_macro'].mean()))
    the_list[2].append((scores['test_recall_macro'].mean()))

    print("Accuracy: %0.4f (+/- %0.2f)" % (scores['test_accuracy'].mean(), scores['test_accuracy'].std() * 2))
    print("Recall: %0.4f (+/- %0.2f)" % (scores['test_recall_macro'].mean(), scores['test_recall_macro'].std() * 2))
    print("Precision: %0.4f (+/- %0.2f)" % (
    scores['test_precision_macro'].mean(), scores['test_precision_macro'].std() * 2))
    print("######")

    #W2V - RFC

    cores = multiprocessing.cpu_count()  # Count the number of cores in a computer

    data = w2v(parseData.fullTrainSet)

    rfc = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=2, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
                           oob_score=False, random_state=0, verbose=0, warm_start=False)

    print("###W2V - RFC###")
    scores = cross_validate(rfc, data, parseData.fullTrainSet['Category'],
                            scoring=scoring,
                            cv=10, return_train_score=False)
    print("######")

    the_list[0].append((scores['test_accuracy'].mean()))
    the_list[1].append((scores['test_precision_macro'].mean()))
    the_list[2].append((scores['test_recall_macro'].mean()))

    print("Accuracy: %0.4f (+/- %0.2f)" % (scores['test_accuracy'].mean(), scores['test_accuracy'].std() * 2))
    print("Recall: %0.4f (+/- %0.2f)" % (scores['test_recall_macro'].mean(), scores['test_recall_macro'].std() * 2))
    print("Precision: %0.4f (+/- %0.2f)" % (scores['test_precision_macro'].mean(), scores['test_precision_macro'].std() * 2))

    #W2V - SVM

    cores = multiprocessing.cpu_count()  # Count the number of cores in a computer

    data = w2v(parseData.fullTrainSet)

    lsvc = LinearSVC(random_state=0, tol=1e-5)

    print("###W2V - SVM###")
    scores = cross_validate(lsvc, data, parseData.fullTrainSet['Category'],
                            scoring=scoring,
                            cv=10, return_train_score=False)
    print("######")

    the_list[0].append((scores['test_accuracy'].mean()))
    the_list[1].append((scores['test_precision_macro'].mean()))
    the_list[2].append((scores['test_recall_macro'].mean()))

    print("Accuracy: %0.4f (+/- %0.2f)" % (scores['test_accuracy'].mean(), scores['test_accuracy'].std() * 2))
    print("Recall: %0.4f (+/- %0.2f)" % (scores['test_recall_macro'].mean(), scores['test_recall_macro'].std() * 2))
    print("Precision: %0.4f (+/- %0.2f)" % (scores['test_precision_macro'].mean(), scores['test_precision_macro'].std() * 2))

    #Beat the benchmark

    textLRClf = Pipeline([
        ('vect', CountVectorizer(min_df=1, stop_words='english')),
        ('tfidf', TfidfTransformer()),
        ('clf', linear_model.SGDClassifier(max_iter=1000, tol=1e-3))

    ])

    print("###My Method###")

    scores = cross_validate(textLRClf, parseData.fullTrainSet['Content'], parseData.fullTrainSet['Category'],
                            scoring=scoring,
                            cv=10, return_train_score=False)

    the_list[0].append((scores['test_accuracy'].mean()))
    the_list[1].append((scores['test_precision_macro'].mean()))
    the_list[2].append((scores['test_recall_macro'].mean()))

    print("Accuracy: %0.4f (+/- %0.2f)" % (scores['test_accuracy'].mean(), scores['test_accuracy'].std() * 2))
    print("Recall: %0.4f (+/- %0.2f)" % (scores['test_recall_macro'].mean(), scores['test_recall_macro'].std() * 2))
    print("Precision: %0.4f (+/- %0.2f)" % (scores['test_precision_macro'].mean(), scores['test_precision_macro'].std() * 2))
    print("######")


    df = pd.DataFrame(the_list, columns=['Statistic Measure', 'Random Forest(BoW)', 'SVM(BoW)', 'SVM(SVD)', 'Random Forest(SVD)', 'SVM (W2V)', 'Random Forest (W2V)','My Method'])
    df.to_csv("/Users/kechagiaskonstantinos/Downloads/Datasets-2018/EvaluationMetric_10fold.csv", sep=',',
              encoding='utf-8',
              index=False)