# Implementing Node2Vec algorithm from scratch and using Logistic Regression to classify the nodes

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.linear_model import LogisticRegression
import networkx as nx
from gensim.models import Word2Vec
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV

class Node2VecAndLogisticRegression:
    def __init__(self):
        self.cora_trainDataFilePath = os.path.join(os.getcwd(), "dataset/cora_train.cites")
        self.cora_testDataFilePath = os.path.join(os.getcwd(), "dataset/cora_test.cites")
        self.coraContentFilePath = os.path.join(os.getcwd(), "dataset/cora.content")
        self.coraTrainData = None
        self.coraTestData = None
        self.coraContent = None
        self.edgelist = None
        self.graph = None
        self.nodeLabelMap = {}
        self.evaluationMetricFileName = 'lr_metrics.txt'

    # Reading the data from the file
    def readData(self):
        # Getting the edge list
        try:
            # Reading both the train and test data
            self.coraTrainData = pd.read_csv(self.cora_trainDataFilePath, sep='\t', header=None, names=["target", "source"])
            self.coraTestData = pd.read_csv(self.cora_testDataFilePath, sep='\t', header=None, names=["target", "source"])
            # Combining the train and test data as edge list
            self.edgelist = pd.concat([self.coraTrainData, self.coraTestData], axis=0)
        except FileNotFoundError:
            print("File not found. Kindly insert the cora.cites in folder dataset. Exiting the program.")
            exit(1)
        assert self.edgelist.shape[1] == 2, "The number of columns in the edgelist is not 2"
        assert self.edgelist.shape[0] == (4343 + 1086), "The number of rows in the edgelist is not 5429"

    # Creating graph from edge list using networkx
    def createGraph(self):
        self.graph = nx.from_pandas_edgelist(self.edgelist, 'source', 'target')
        # Checking if the number of nodes in the graph is 2708 (Without test data it is 2596)
        assert self.graph.number_of_nodes() == 2708, "The number of nodes in the graph is not 2708"
        return self.graph
    
    # Reading the content file
    def readContentFile(self):
        try:
            self.coraContent = pd.read_csv(self.coraContentFilePath, sep='\t', header=None)
        except FileNotFoundError:
            print("File not found. Kindly insert the cora.content in folder dataset. Exiting the program.")
            exit(1)
        assert self.coraContent.shape[1] == 1435, "The number of columns in the content file is not 1435"
        assert self.coraContent.shape[0] == 2708, "The number of rows in the content file is not 2708"

    # Creating the node label map
    def createNodeLabelMap(self):
        for index, row in self.coraContent.iterrows():
            self.nodeLabelMap[row[0]] = row[1434]

    # Initializing the probabilities dictionary
    def initializeProbabilities(self, graph):
        probabilities = {}
        for node in graph.nodes():
            if node not in probabilities:
                probabilities[node] = {}
            if 'probability' not in probabilities[node]:
                probabilities[node]['probability'] = {}
        return probabilities

    # Helper function to get the probablity value
    def getProbalityValue(self, sourceNode, destinationNode, graph, p, q):
        if graph.has_edge(sourceNode, destinationNode) or graph.has_edge(destinationNode, sourceNode):
            return 1
        elif nx.common_neighbors(graph, sourceNode, destinationNode):
            return 1/p
        else:
            return 1/q
    
    # Calculating probabilities for the nodes
    def getProbablities(self, graph, p, q, probabilities):
        for sourceNode in graph.nodes():
            for currentNode in graph.neighbors(sourceNode):
                probablity = []
                for destinationNode in graph.neighbors(currentNode):
                    probablity.append(self.getProbalityValue(sourceNode, destinationNode, graph, p, q))
                probabilities[sourceNode]['probability'][currentNode] = probablity / np.sum(probablity)
        return probabilities
    
    # Helper function for random walk
    def randomWalk(self, graph, probabilities, walkLength, walk):
        for j in range(walkLength):
            currentNode = walk[-1]
            if len(walk) > 1:
                previousNode = walk[-2]
            else:
                previousNode = None
            
            if len(list(graph.neighbors(currentNode))) > 0:
                if previousNode is None:
                    nextNode = np.random.choice(list(graph.neighbors(currentNode)), 1)[0]
                else:
                    nextNode = np.random.choice(list(graph.neighbors(currentNode)), 1, p=probabilities[previousNode]['probability'][currentNode])[0]
            
            walk.append(nextNode)

        return walk

    # Making random walks
    def randomWalks(self, graph, probabilities, walkLength):
        randonWalks = []
        for node in graph.nodes():
            for i in range(walkLength):
                walk = [node]
                walk = self.randomWalk(graph, probabilities, walkLength, walk)
                randonWalks.append(walk)
        return randonWalks
    
    # Generating probabilities for the nodes
    def generateProbabilities(self, graph, p, q):
        probabilities = self.initializeProbabilities(graph)
        probabilities = self.getProbablities(graph, p, q, probabilities)
        return probabilities
    
    # Getting random walks
    def getRandomWalks(self, graph, probabilities, walkLength):
        walks = self.randomWalks(graph, probabilities, walkLength)
        return walks
    
    # creating Node2Vec model
    def createNode2VecModel(self, walks, dimensions, windowSize, iterations):
        model = Word2Vec(sentences=walks, vector_size=dimensions, window=windowSize, min_count=0, sg=1, workers=4, epochs=iterations)
        return model.wv
    
    # Generate and get the embeddings
    def generateEmbeddings(self, graph, p, q, walkLength, dimensions, windowSize, iterations):
        probabilities = self.generateProbabilities(graph, p, q)
        walks = self.getRandomWalks(graph, probabilities, walkLength)
        assert len(walks) == 2708 * walkLength, "The number of walks is not 2708 * " + str(walkLength)

        embeddings = self.createNode2VecModel(walks, dimensions, windowSize, iterations)
        assert len(embeddings) == 2708, "The number of embeddings is not 2708"

        return embeddings
    
    # Getting the embeddings for the nodes
    def getEmbeddings(self, graph, p, q, walkLength, dimensions, windowSize, iterations):
        embeddings = self.generateEmbeddings(graph, p, q, walkLength, dimensions, windowSize, iterations)
        return embeddings
    
    # Preparing data for logistic regression
    def prepareDataForLogisticRegression(self, embeddings):
        # Train data is the cora_train.cites and test data is the cora_test.cites
        # Getting the train and test data. Only consider the source nodes
        trainData = self.coraTrainData['source'].values
        testData = self.coraTestData['source'].values

        # Getting the labels for the train and test data
        trainLabels = []
        testLabels = []

        for node in trainData:
            trainLabels.append(self.nodeLabelMap[node])

        for node in testData:
            testLabels.append(self.nodeLabelMap[node])

        # Getting the vectors for the train and test data
        trainVectors = []
        testVectors = []

        for node in trainData:
            trainVectors.append(embeddings[node])

        for node in testData:
            testVectors.append(embeddings[node])

        return trainVectors, trainLabels, testVectors, testLabels
    
    # Training the logistic regression model
    def trainLR(self, trainVectors, trainLabels):
        # Finding the best parameters for Logistic Regression
        models = {
            'LR': LogisticRegression()
        }

        # Hyperparameters for Logistic Regression
        logistic_params = {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        }

        # Model list for hyperparameter tuning
        model_list = [
            ('LR', models['LR'], logistic_params)
        ]

        modelParams = {}

        for model_name, model, params in model_list:
            randomSearch = RandomizedSearchCV(estimator=model,
                                            param_distributions=params,
                                            n_iter=100,
                                            cv=3,
                                            verbose=2,
                                            n_jobs=-1)
            # combine the training and validation set
            randomSearch.fit(trainVectors, trainLabels)
            modelParams[model_name] = randomSearch.best_params_

        # Displaying the best parameters for Logistic Regression
        for model_name, params in modelParams.items():
            print(model_name)
            print(params)
            print("\n\n\n")

        # Training the Logistic Regression with the best parameters
        model = models['LR']
        model.set_params(**modelParams['LR'])
        model.fit(trainVectors, trainLabels)

        return model
    
    # Testing the logistic regression model
    def testLR(self, model, testVectors, testLabels):
        predictions = model.predict(testVectors)
        accuracy = accuracy_score(testLabels, predictions)
        precision = precision_score(testLabels, predictions, average='weighted')
        recall = recall_score(testLabels, predictions, average='weighted')
        f1 = f1_score(testLabels, predictions, average='macro')
        confusionMatrix = confusion_matrix(testLabels, predictions)

        return accuracy, precision, recall, f1, confusionMatrix
    
    # Saving the evaluation metrics in file "lr_metrics.txt"
    def saveEvaluationMetrics(self, accuracy, precision, recall, f1, confusionMatrix):
        with open(self.evaluationMetricFileName, 'w') as file:
            file.write("Accuracy: " + str(accuracy) + "\n")
            file.write("Precision: " + str(precision) + "\n")
            file.write("Recall: " + str(recall) + "\n")
            file.write("F1: " + str(f1) + "\n")
            file.write("Confusion Matrix: \n" + str(confusionMatrix) + "\n")

    # driver function
    def driverFunction(self):
        p = 1
        q = 1
        walkLength = 50
        dimensions = 64
        windowSize = 10
        iterations = 10

        self.readData()
        self.createGraph()
        self.readContentFile()
        self.createNodeLabelMap()

        embeddings = self.getEmbeddings(self.graph, p, q, walkLength, dimensions, windowSize, iterations)
        print("Embeddings generated")
        trainVectors, trainLabels, testVectors, testLabels = self.prepareDataForLogisticRegression(embeddings)
        model = self.trainLR(trainVectors, trainLabels)
        accuracy, precision, recall, f1, confusionMatrix = self.testLR(model, testVectors, testLabels)
        self.saveEvaluationMetrics(accuracy, precision, recall, f1, confusionMatrix)
        print("Evaluation metrics saved")

if __name__ == "__main__":
    node2vecAndLR = Node2VecAndLogisticRegression()
    node2vecAndLR.driverFunction()