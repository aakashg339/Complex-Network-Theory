import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

METRIC_FILE_NAME = 'gcn_metrics.txt'
NUMBER_OF_HIDDEN_UNITS = 16
DROP_OUT = 0.5
LEARNING_RATE = 0.01
NUMBER_OF_EPOCHS = 100

# Defining the GCN Layer
class GCNLayer(nn.Module):
    def __init__(self, inputFeatures, outputFeatures):
        super(GCNLayer, self).__init__()
        self.linearLayer = nn.Linear(inputFeatures, outputFeatures, bias=False)
    
    def forward(self, input, adjacencyMatrix):
        # getting the degree matrix
        # degree matrix is a diagonal matrix where the diagonal elements are the sum of the corresponding row of the adjacency matrix, raised to the power of -0.5
        degreeDiagonal = torch.sum(adjacencyMatrix, dim=1)
        degreeDiagonalNegativeSqrt = degreeDiagonal.pow(-0.5)
        degreeMatrix = torch.diag_embed(degreeDiagonalNegativeSqrt)

        # calcualting D^-0.5 * A * D^-0.5 * X, where X is the feature matrix(here input)
        intermidateResult1 = torch.matmul(degreeMatrix, adjacencyMatrix)
        intermidateResult2 = torch.matmul(intermidateResult1, degreeMatrix)
        intermidateResult3 = torch.matmul(intermidateResult2, input)

        # passing the result through a linear layer or multiplying with a weight matrix
        layerResult = self.linearLayer(intermidateResult3)
        return layerResult

# Defining the GCN Network
class GCNNetwork(nn.Module):
    def __init__(self, numberOfFeatures, numberOfHidenUnits, numberOfClasses, dropout=0.5):
        super(GCNNetwork, self).__init__()
        self.gcnLayer1 = GCNLayer(numberOfFeatures, numberOfHidenUnits)
        self.gcnLayer2 = GCNLayer(numberOfHidenUnits, numberOfClasses)
        self.dropout = dropout

    def forward(self, featureMatrix, adjacencyMatrix):
        x = self.gcnLayer1(featureMatrix, adjacencyMatrix)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcnLayer2(x, adjacencyMatrix)
        x = F.log_softmax(x, dim=1)
        return x

# Returning edges as source and destination indices by reading as <destination>\t<source>
def loadDataAndMapToIndex(dataFilePath, idIndexMap):
    edges = pd.read_csv(dataFilePath, header=None, sep='\t').values
    # print(edges[:5])
    edgesAsIds = np.zeros(edges.shape, dtype=np.int32)
    for i, edge in enumerate(edges):
        edgesAsIds[i, 0] = idIndexMap[edge[1]]
        edgesAsIds[i, 1] = idIndexMap[edge[0]]
    # # display the first 5 rows
    # print(edgesAsIds[:5])
    return edgesAsIds

def getWordAttributes(coraContentDataframe):
    wordAttributesNumpyForm = np.array(coraContentDataframe.iloc[:, 1:-1], dtype=np.float32)
    return wordAttributesNumpyForm

def getEncodedLabels(coraContentDataframe):
    uniqueLabels = coraContentDataframe.iloc[:, -1].unique()
    labelIndexMap = {}
    for i, label in enumerate(uniqueLabels):
        labelIndexMap[label] = i
    encodedLabels = coraContentDataframe.iloc[:, -1].map(labelIndexMap).values
    return encodedLabels

def getIdIndexMap(coraContentDataframe):
    ids = np.array(coraContentDataframe[0], dtype=np.int32)
    idIndexMap = {}
    for i, id in enumerate(ids):
        idIndexMap[id] = i
    return idIndexMap

def getAdjacencyMatrix(allEdges, numberOfNodes):
    adjacencyMatrix = np.zeros((numberOfNodes, numberOfNodes), dtype=np.float32)
    adjacencyMatrix[allEdges[:, 0], allEdges[:, 1]] = 1

    # Adding self-loops
    adjacencyMatrix += np.eye(numberOfNodes)

    return adjacencyMatrix

def getCoraData(coraContentDatapath, trainDatapath, testDatapath):
    coraContentDataframe = pd.read_csv(coraContentDatapath, header=None, sep='\t')

    # Extracting the word attributes in tensor form
    wordAttributesNumpyForm = getWordAttributes(coraContentDataframe)
    wordAttributes = torch.FloatTensor(wordAttributesNumpyForm)
    
    # Labels encoded manually
    encodedLabels = getEncodedLabels(coraContentDataframe)
    labels = torch.LongTensor(encodedLabels)
    
    # Creating a map of node id to index
    idIndexMap = getIdIndexMap(coraContentDataframe)

    # Getting the data for training and testing
    trainEdgesAsIds = loadDataAndMapToIndex(trainDatapath, idIndexMap)
    testEdgesAsIds = loadDataAndMapToIndex(testDatapath, idIndexMap)
    # allEdges = trainEdgesAsIds
    allEdges = np.vstack((trainEdgesAsIds, testEdgesAsIds))

    # Getting the adjacency matrix
    adjacencyMatrix = getAdjacencyMatrix(allEdges, labels.shape[0])
    adjacencyMatrix = torch.FloatTensor(adjacencyMatrix)

    # Getting the unique train and test ids
    trainIds = np.unique(trainEdgesAsIds)
    testIds = np.unique(testEdgesAsIds)

    return wordAttributes, adjacencyMatrix, labels, trainIds, testIds

# Model Training and Evaluation
def trainModel(model, optimizer, features, adjacencyMatrix, labels, idsTrain, epochs=200):
    lossFunction = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adjacencyMatrix)
        loss = lossFunction(output[idsTrain], labels[idsTrain])
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f'Epoch {epoch}: Training loss {loss.item()}')

# Evaluate the model on the test set. Print the loss, accuracy, precision, recall, and F1 score and confusion matrix
def evaluateModel(model, features, adjacencyMatrix, labels, idsTest):
    model.eval()
    with torch.no_grad():
        output = model(features, adjacencyMatrix)
        prediction = output[idsTest].max(1)[1].type_as(labels[idsTest])
        labelsTest = labels[idsTest].cpu().numpy()
        predictionTest = prediction.cpu().numpy()
        
        accuracy = accuracy_score(labelsTest, predictionTest)
        precision = precision_score(labelsTest, predictionTest, average='weighted')
        recall = recall_score(labelsTest, predictionTest, average='weighted')
        f1 = f1_score(labelsTest, predictionTest, average='macro')
        confusionMatrix = confusion_matrix(labelsTest, predictionTest)
        
        return accuracy, precision, recall, f1, confusionMatrix
    
def saveMetrics(accuracy, precision, recall, f1, confusionMatrix, filename):
    with open(filename, 'w') as file:
        file.write(f'Accuracy: {accuracy}\n')
        file.write(f'Precision: {precision}\n')
        file.write(f'Recall: {recall}\n')
        file.write(f'F1: {f1}\n')
        file.write(f'Confusion Matrix:\n{confusionMatrix}\n')

if __name__ == '__main__':
    coraContentDatapath = '../dataset/cora.content'
    coraTrainDatapath = '../dataset/cora_train.cites'
    coraTestDatapath = '../dataset/cora_test.cites'

    features, adjacencyMatrix, labels, trainIds, testIds = getCoraData(coraContentDatapath, coraTrainDatapath, coraTestDatapath)
    print('Data Loaded')

    #nfeat=features.shape[1], nhid=16, nclass=len(torch.unique(labels)), dropout=0.5)
    GCNmodel = GCNNetwork(numberOfFeatures=features.shape[1], numberOfHidenUnits=NUMBER_OF_HIDDEN_UNITS, numberOfClasses=labels.max().item() + 1, dropout=DROP_OUT)
    optimizer = optim.Adam(GCNmodel.parameters(), lr=LEARNING_RATE)

    trainModel(GCNmodel, optimizer, features, adjacencyMatrix, labels, torch.LongTensor(trainIds), epochs=NUMBER_OF_EPOCHS)
    print('Model Trained')
    
    accuracy, precision, recall, f1, confusionMatrix = evaluateModel(GCNmodel, features, adjacencyMatrix, labels, torch.LongTensor(testIds))

    saveMetrics(accuracy, precision, recall, f1, confusionMatrix, METRIC_FILE_NAME)
    print('Metrics saved')