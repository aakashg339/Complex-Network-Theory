
# Web scraping system with Python (with multithreading)

Program to scrape downloaded HTML web pages of investing.com (The folder containing the HTML web pages was not included to avoid any license violations). Find below one example whose HTML can be downloaded and used. 

https://www.investing.com/equities/apple-computer-inc

This project is part of a Computing Lab (CS69201) End Semester examination at IIT Kharagpur. (A PDF of the assignment can be found in the repository)

#### Programming Languages Used
* Python
* C

#### Libraries Used
* bs4 (Python)
* sys (Python)
* pthread (C)
In case of any missing library, kindly install it using 
    - pip3 install < library name > (for Python)
(Some libraries mentioned above come as part of python3 or C)

### Role of parser.py 
It takes the HTML file to be parsed as input, gets the company name and stock price, and stores it in the output file (whose name is provided as input to the Python program).

### Role of driver.c
The working of the driver.c is explained in the steps below.
1. Read all the HTML file names in the directory http_dump/dump_http/files (relative to the current directory).
2. Return to the current directory where the Python file is present.
3. Use multithreading, where each thread calls parser.py to extract the required data from the file. There is one thread for each HTML file. The output of the Python program is then read and stored in 'parsed_data.log.'
4. Read 'parsed_data.log,' get the company's details with the highest stock price, and display it.

## Running it locally on your machine

1. Clone this repository and cd to the project root.
2. Download HTML files of equities pages of different companies (such as https://www.investing.com/equities/align-technology).
3. Compile the C program using the command - gcc driver.c -o < name of the program > -lpthread
4. Run the program using - ./< name of the program >
## Purpose

The project was used to understand the usage of threads.


    
    # Calculating all pair shortest path with path length, number of shortest paths, and the nodes in the path.
    # The output is stored as a dictionary of dictionaries.
    # Output format: {source paper id: {target paper id: [path length, number of shortest paths, list of list of nodes where each list is a path]}}
    # below method uses single source shortest path algorithm to record the path length, number of shortest paths and the nodes in the path.
    

    # Find all shortest path from source to destination
    # Reference - https://www.geeksforgeeks.org/print-all-shortest-paths-between-given-source-and-destination-in-an-undirected-graph/
    # Call below and store as {<source research paper id> : [<target research paper id> : [list of nodes in the path]]}}

    # Code to calculate closeness centrality, betweenness centrality and pagerank for the cora dataset without using networkx library.
# The CORA graph can be considered as undirected.
# cora.cites format: <target research paper id> <source research paper id>
# As research paper ids are not continuous, we will use adjacency list to represent the graph.
# Graph visualization https://graphonline.ru/en/

# From net example to find distance
0-1
0-2
1-3
1-4
2-3
3-5
4-5

# From book to calculate centrality
a-b
b-c
c-d
d-e
d-h
e-h
e-g
e-f
f-g
g-h

# From ppt to calculate centrality
b-a
c-b
d-b
e-c
e-d

# From PPT for page rank
y-y
a-y
y-a
m-a
a-m