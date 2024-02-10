import numpy as np
import pandas as pd
import os
from collections import deque
from sys import maxsize


class CentralityMeasuresAndPageRank:

    def __init__(self, currentDirectoryPath):
        self.dataFilePath = os.path.join(currentDirectoryPath, "cora")
        self.outputFilePath = os.path.join(currentDirectoryPath, "centralities")
        self.edgelist = None
        self.adjacencyList = {}
        self.vertexList = []
        self.numberOfNodes = 0
        self.outDegreeOfNodes = {}
        self.inDegreeOfNodes = {}
        self.allPairPathsData = {}
        self.closenessCentrality = {}
        self.betweennessCentrality = {}
        self.incommingEdges = {}

    # Reading the data from the file
    def readData(self):
        # Getting the edge list
        try:
            self.edgelist = pd.read_csv(os.path.join(self.dataFilePath, "cora.cites"), sep='\t', header=None, names=["target", "source"])
        except FileNotFoundError:
            print("File not found. Kindly insert the cora.cites in folder cora. Exiting the program.")
            exit(1)
        assert self.edgelist.shape[1] == 2, "The number of columns in the edgelist is not 2"
        assert self.edgelist.shape[0] == 5429, "The number of rows in the edgelist is not 5429"
    
    # Function to add the edge to the incoming edges
    def addEdgeToIncomingEdges(self, source, target):
        if target not in self.incommingEdges:
            self.incommingEdges[target] = [source]
        else:
            self.incommingEdges[target].append(source)
    
    # Creating the adjacency list
    def createAdjacencyList(self):
        for index, row in self.edgelist.iterrows():
            if row["source"] not in self.adjacencyList:
                self.adjacencyList[row["source"]] = [row["target"]]
            else:
                self.adjacencyList[row["source"]].append(row["target"])
            self.addEdgeToIncomingEdges(row["source"], row["target"])

            # Adding the vertices to the nodes list
            if row["source"] not in self.vertexList:
                self.vertexList.append(row["source"])
            if row["target"] not in self.vertexList:
                self.vertexList.append(row["target"])

        # Adding the nodes which are not in the adjecency list but present in vertex list
        for node in self.vertexList:
            if node not in self.adjacencyList:
                self.adjacencyList[node] = []
        
        # Sort the adjacency list as per keys
        self.adjacencyList = dict(sorted(self.adjacencyList.items()))
        self.vertexList.sort()
        
        assert list(self.adjacencyList.keys()) == self.vertexList, "The adjacency list and vertex list are not in the same order"
        
        self.numberOfNodes = len(self.vertexList)
        assert self.numberOfNodes == 2708, "The number of nodes in the graph is not 2708"

    # Function to calculate the degree of the nodes
    def degreeOfNode(self):
        for node in self.vertexList:
            if node in self.adjacencyList:
                self.outDegreeOfNodes[node] = len(self.adjacencyList[node])
            else:
                # If the node is not in the adjacency list then add edges for every incoming edge. This will be handled later
                self.outDegreeOfNodes[node] = 0
                
            self.inDegreeOfNodes[node] = 0
        for node in self.vertexList:
            if node in self.adjacencyList:
                for destination in self.adjacencyList[node]:
                    self.inDegreeOfNodes[destination] += 1
                    
    # Function to adjust the graph for nodes with no outgoing edges
    def adjustGraphForNodesWithNoOutgoingEdges(self):
        for node in self.vertexList:
            if len(self.adjacencyList[node]) == 0:
                # For every edge coming to the node, add the node to the adjacency list
                #self.adjacencyList[node] = []
                for source in self.incommingEdges[node]:
                    self.adjacencyList[node].append(source)

    # Function to finds all the paths from source to destination
    def findPathsSourceToDestination(self, pathList, path, parentDict, currentNode):
        # When we reach destination, we insert the path in pathList
        if (currentNode == -1):
            pathList.append(path.copy())
            return

        # Loop for all the parents of the given vertex, to find all the paths
        for parentU in parentDict[currentNode]:

            path.append(currentNode)

            self.findPathsSourceToDestination(pathList, path, parentDict, parentU)

            # Removing current vertex to back track
            path.pop()

    # Function to run bfs and find the shortest paths from source to destination
    def bfs(self, parentDict, startNode):

        # Dictionary to store the distance of each node from the source
        distanceMap = {}

        for node in self.vertexList:
            distanceMap[node] = maxsize

        q = deque()

        q.append(startNode)
        parentDict[startNode] = [-1]
        distanceMap[startNode] = 0

        # Looping until queue is empty
        while q:
            currentNode = q[0]
            q.popleft()
            if currentNode in self.adjacencyList:
                for adjNode in self.adjacencyList[currentNode]:
                    if (distanceMap[adjNode] > distanceMap[currentNode] + 1):
                        # As a shorter path is found, update the distance and add the node to the queue
                        distanceMap[adjNode] = distanceMap[currentNode] + 1
                        q.append(adjNode)
                        parentDict[adjNode].clear()
                        parentDict[adjNode].append(currentNode)

                    elif (distanceMap[adjNode] == distanceMap[currentNode] + 1):
                        # Another path of the same length is found
                        parentDict[adjNode].append(currentNode)
    
    # Function to find all the shortest paths from source to destination
    def shortestPathsFromSourceToDestination(self, source):
        # List to store the paths
        parentDict = {}

        for node in self.vertexList:
            parentDict[node] = []

        # Function call to bfs
        self.bfs(parentDict, source)

        # Considering all nodes except itself
        for node in self.vertexList:
            
            pathList = []
            path = []

            if node == source:
                continue
            if source not in self.allPairPathsData:
                self.allPairPathsData[source] = {}
            
            self.allPairPathsData[source][node] = {'numberOfPaths': 0,
                                                  'pathLength': -1,
                                                  'paths':  []}

            # Function call to findPathsSourceToDestination
            self.findPathsSourceToDestination(pathList, path, parentDict, node)
            for eachPath in pathList:
                # Reversing the path to get the correct path
                eachPath = eachPath[::-1]

                self.allPairPathsData[source][node]['numberOfPaths'] += 1
                self.allPairPathsData[source][node]['pathLength'] = len(eachPath) - 1
                self.allPairPathsData[source][node]['paths'].append(eachPath.copy())
                        
    # Function to find the closeness centrality of the nodes
    def closenessCentralityMeasure(self):
        # If a node is not reachable from the source then take a very large value as the path length, to signify that the node is not reachable
        self.closenessCentrality = {}
        for node in self.vertexList:
            self.closenessCentrality[node] = 0
            for destination in self.allPairPathsData[node]:
                if self.allPairPathsData[node][destination]['pathLength'] != -1:
                    self.closenessCentrality[node] += self.allPairPathsData[node][destination]['pathLength']
                else:
                    self.closenessCentrality[node] += self.numberOfNodes
            if self.closenessCentrality[node] != 0:
                self.closenessCentrality[node] = (self.numberOfNodes - 1) / self.closenessCentrality[node]
        
        # writing result to file
        self.writeResultToFile(self.closenessCentrality, "closeness.txt")
        print("completed closeness centrality")

    def betweennessCentralityMeasure(self):
        self.betweennessCentrality = {}
        for node in self.vertexList:
            # pairsCompleted = set()
            self.betweennessCentrality[node] = 0
            for source in self.vertexList:
                if source == node:
                    continue
                for destination in self.allPairPathsData[source]:
                    # if destination == node or (source, destination) in pairsCompleted or self.allPairPathsData[source][destination]['pathLength'] == -1:
                    if destination == node or self.allPairPathsData[source][destination]['pathLength'] == -1:
                        continue
                    shortestPathsIncludingNode = 0
                    for path in self.allPairPathsData[source][destination]['paths']:
                        if node in path:
                            shortestPathsIncludingNode += 1
                    self.betweennessCentrality[node] += shortestPathsIncludingNode / self.allPairPathsData[source][destination]['numberOfPaths']
                    # pairsCompleted.add((source, destination))
                    # pairsCompleted.add((destination, source))
            # print("Betweenness centrality for node " + str(node) + " is " + str(self.betweennessCentrality[node]))

        # Normalize the betweenness centrality
        for node in self.betweennessCentrality:
            self.betweennessCentrality[node] = self.betweennessCentrality[node] / ((self.numberOfNodes - 1) * (self.numberOfNodes - 2))
        

        # writing result to file
        self.writeResultToFile(self.betweennessCentrality, "betweenness.txt")
        print("completed betweenness centrality")

    # Function to find the pagerank of the nodes using power iteration method
    def pageRank(self):
        # Creating the transformation matrix where cell i -> j represents the probability of going from node i to node j
        # Using pandas to create the dataframe where the index and columns are named as per node id
        transformationMatrix = pd.DataFrame(index=self.vertexList, columns=self.vertexList)
        # Filling the dataframe with 0
        transformationMatrix.fillna(0, inplace=True)
        # Filling the dataframe with the probability of going from node i to node j
        for src in self.vertexList:
            if src in self.adjacencyList:
                for dest in self.adjacencyList[src]:
                    if self.outDegreeOfNodes[dest] != 0:
                        transformationMatrix.at[src, dest] = 1 / self.outDegreeOfNodes[dest]
        
        # teleportation probability
        alpha = 0.8

        # teleportation matrix
        teleportationMatrix = pd.DataFrame(index=self.vertexList, columns=self.vertexList)
        teleportationMatrix.fillna(0, inplace=True)
        teleportationMatrix = teleportationMatrix + (1 - alpha) / self.numberOfNodes

        # matrix M
        matrixM = alpha * transformationMatrix + teleportationMatrix

        # initial page rank
        initialPageRank = pd.DataFrame(index=self.vertexList, columns=['pageRank'])
        initialPageRank.fillna(1 / self.numberOfNodes, inplace=True)
        initialPageRank = initialPageRank.round(6)

        # To store the calculated page rank
        pageRank = pd.DataFrame(index=self.vertexList, columns=['pageRank'])
        pageRank.fillna(0, inplace=True)
        pageRank = initialPageRank.copy()

        # power iteration method
        # Set a threshold as 0.000001
        epsilon = 0.000001
        for i in range(1, 51):
            # print("Iteration " + str(i))
            newPageRank = matrixM.dot(pageRank)

            # Check for convergence
            if ((newPageRank - pageRank).abs() < epsilon).all().all():
                print(f"Converged at iteration {i}")
                break

            pageRank = newPageRank

        # Scaling the page rank values
        # maxPageRank = pageRank['pageRank'].max()
        # minPageRank = pageRank['pageRank'].min()
        # pageRank['pageRank'] = (pageRank['pageRank'] - minPageRank) / (maxPageRank - minPageRank)
        
        # print(pageRank)
        
        # Creating mapping of node id with page rank
        pageRankMapping = {}
        for index, row in pageRank.iterrows():
            pageRankMapping[index] = row['pageRank']
        
        # writing result to file
        self.writeResultToFile(pageRankMapping, "pagerank.txt")
        print("completed page rank")

    # Function to write result to file by rounding reach value to 6 decimal places
    def writeResultToFile(self, result, fileName):
        # sort the result as per value in descending order
        result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))

        # Check if the directory exists
        if not os.path.exists(self.outputFilePath):
            os.makedirs(self.outputFilePath)

        # Write the result to the file
        with open(os.path.join(self.outputFilePath, fileName), "w") as file:
            for node in result:
                file.write(str(node) + " " + str(round(result[node], 6)) + "\n")

if __name__ == "__main__":
    currentDirectoryPath = os.path.join(os.getcwd())
    centralityMeasuresAndPageRank = CentralityMeasuresAndPageRank(currentDirectoryPath)
    centralityMeasuresAndPageRank.readData()
    centralityMeasuresAndPageRank.createAdjacencyList()
    centralityMeasuresAndPageRank.adjustGraphForNodesWithNoOutgoingEdges()
    centralityMeasuresAndPageRank.degreeOfNode()
    print("Preprocessing done")
    # Calling using different source
    for node in centralityMeasuresAndPageRank.vertexList:
        centralityMeasuresAndPageRank.shortestPathsFromSourceToDestination(node)
    
    print("Creation of shortest path list done")
    centralityMeasuresAndPageRank.closenessCentralityMeasure()
    centralityMeasuresAndPageRank.betweennessCentralityMeasure()
    centralityMeasuresAndPageRank.pageRank()