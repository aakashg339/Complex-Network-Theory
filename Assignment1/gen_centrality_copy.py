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
        self.edgelist = pd.read_csv(os.path.join(self.dataFilePath, "cora.cites"), sep='\t', header=None, names=["target", "source"])
        assert self.edgelist.shape[1] == 2, "The number of columns in the edgelist is not 2"
        assert self.edgelist.shape[0] == 5429, "The number of rows in the edgelist is not 5429"
    
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
        
        # Sort the adjacency list as per keys
        self.adjacencyList = dict(sorted(self.adjacencyList.items()))
        self.vertexList.sort()
        
        self.numberOfNodes = len(self.vertexList)
        assert self.numberOfNodes == 2708, "The number of nodes in the graph is not 2708"

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
            if node not in self.adjacencyList:
                # For every edge coming to the node, add the node to the adjacency list
                self.adjacencyList[node] = []
                for source in self.incommingEdges[node]:
                    self.adjacencyList[node].append(source)

    # Function which finds all the paths
    # and stores it in paths array
    def find_paths(self, paths, path, parent, u):
        # Base Case
        if (u == -1):
            paths.append(path.copy())
            return

        # Loop for all the parents
        # of the given vertex
        for par in parent[u]:

            # Insert the current
            # vertex in path
            path.append(u)

            # Recursive call for its parent
            self.find_paths(paths, path, parent, par)

            # Remove the current vertex
            path.pop()

    # Function which performs bfs
    # from the given source vertex
    def bfs(self, parent, start):

        # dist will contain shortest distance
        # from start to every other vertex
        dist = {}

        for node in self.vertexList:
            dist[node] = maxsize

        q = deque()

        # Insert source vertex in queue and make
        # its parent -1 and distance 0
        q.append(start)
        parent[start] = [-1]
        dist[start] = 0

        # Until Queue is empty
        while q:
            u = q[0]
            q.popleft()
            if u in self.adjacencyList:
                for v in self.adjacencyList[u]:
                    if (dist[v] > dist[u] + 1):

                        # A shorter distance is found
                        # So erase all the previous parents
                        # and insert new parent u in parent[v]
                        dist[v] = dist[u] + 1
                        q.append(v)
                        parent[v].clear()
                        parent[v].append(u)

                    elif (dist[v] == dist[u] + 1):

                        # Another candidate parent for
                        # shortes path found
                        parent[v].append(u)
    
    # Function to find all the shortest paths from source to destination
    def shortestPathsFromSourceToDestination(self, source):
        # List to store the paths
        parent = {}

        for node in self.vertexList:
            parent[node] = []

        # Function call to bfs
        self.bfs(parent, source)

        # Considering all nodes except itself
        for node in self.vertexList:
            
            paths = []
            path = []

            if node == source:
                continue
            if source not in self.allPairPathsData:
                self.allPairPathsData[source] = {}
            
            self.allPairPathsData[source][node] = {'numberOfPaths': 0,
                                                  'pathLength': -1,
                                                  'paths':  []}

            # Function call to find_paths
            self.find_paths(paths, path, parent, node)
            for v in paths:
                # Since paths contain each
                # path in reverse order,
                # so reverse it
                v = v[::-1]
                # Print node for the current path
                # for u in v:
                #     print(u, end = " ")
                # print()

                self.allPairPathsData[source][node]['numberOfPaths'] += 1
                self.allPairPathsData[source][node]['pathLength'] = len(v) - 1
                self.allPairPathsData[source][node]['paths'].append(v.copy())
                        
    # Function to find the closeness centrality of the nodes
    def closenessCentralityMeasure(self):
        self.closenessCentrality = {}
        for node in self.vertexList:
            self.closenessCentrality[node] = 0
            for destination in self.allPairPathsData[node]:
                if self.allPairPathsData[node][destination]['pathLength'] != -1:
                    self.closenessCentrality[node] += self.allPairPathsData[node][destination]['pathLength']
            if self.closenessCentrality[node] != 0:
                self.closenessCentrality[node] = (self.numberOfNodes - 1) / self.closenessCentrality[node]
        
        # writing result to file
        self.writeResultToFile(self.closenessCentrality, "closeness.txt")
        print("completed closeness centrality")

    def betweennessCentralityMeasure(self):
        self.betweennessCentrality = {}
        pairsCompleted = []
        for node in self.vertexList:
            pairsCompleted = set()
            self.betweennessCentrality[node] = 0
            for source in self.vertexList:
                if source == node:
                    continue
                for destination in self.allPairPathsData[source]:
                    if destination == node or (source, destination) in pairsCompleted or self.allPairPathsData[source][destination]['pathLength'] == -1:
                        continue
                    shortestPathsIncludingNode = 0
                    for path in self.allPairPathsData[source][destination]['paths']:
                        if node in path:
                            shortestPathsIncludingNode += 1
                    self.betweennessCentrality[node] += shortestPathsIncludingNode / self.allPairPathsData[source][destination]['numberOfPaths']
                    pairsCompleted.add((source, destination))
                    pairsCompleted.add((destination, source))
            print("Betweenness centrality for node " + str(node) + " is " + str(self.betweennessCentrality[node]))

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

        # page rank.
        pageRank = pd.DataFrame(index=self.vertexList, columns=['pageRank'])
        pageRank.fillna(0, inplace=True)
        pageRank = initialPageRank.copy()

        # power iteration method
        epsilon = 1e-6  # Set a small threshold for convergence
        for i in range(1, 21):
            # print("Iteration " + str(i))
            newPageRank = matrixM.dot(pageRank)

            # Check for convergence
            if ((newPageRank - pageRank).abs() < epsilon).all():
                print(f"Converged at iteration {i}")
                break

            pageRank = newPageRank

        # Scaling the page rank values
        maxPageRank = pageRank['pageRank'].max()
        minPageRank = pageRank['pageRank'].min()
        pageRank['pageRank'] = (pageRank['pageRank'] - minPageRank) / (maxPageRank - minPageRank)
        
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
        # print("Shortest paths from source " + str(node) + ":\n")
        centralityMeasuresAndPageRank.shortestPathsFromSourceToDestination(node)
    
    print("Creation of shortest path list done")
    centralityMeasuresAndPageRank.closenessCentralityMeasure()
    # centralityMeasuresAndPageRank.betweennessCentralityMeasure()
    centralityMeasuresAndPageRank.pageRank()
    


        