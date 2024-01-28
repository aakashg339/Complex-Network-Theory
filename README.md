
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