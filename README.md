# EARIN EX8
The goal of this project was to create a program solving taxi problem using Q-Learning technique.  
Created in python using ```gym``` library by Oskar Hącel & Marcin Lisowski  

Politechnika Warszawska, 06.2021

## Preparation
1. Install required python packages with
```cmd
pip install -r requirements.txt
 ```
## Running
Run program with following parameters
```cmd
py ex8.py -l <learning_rate> -e <epsilon> -g <gamma> -i <iterations>
 ```
 ### Example run
 ```cmd
py ex8.py -l 0.1 -e 0.01 -g 0.4 -i 100000
 ```