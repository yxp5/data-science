# Author: yxp5
# Best moment for random choice

import random
import matplotlib.pyplot as plt
import numpy as np

def pick(stopPoint, arr):
    # Corner cases
    if stopPoint in [0, 99]: return arr[stopPoint]
    
    # Get the maximum number before stopPoint
    skippedMax = -1
    for num in arr[:stopPoint]:
        if num > skippedMax: skippedMax = num
    
    # Output the first number greater or equal to skippedMax OR the maximum number after stopPoint
    outputMax = -1
    for num in arr[stopPoint:]:
        if num > outputMax: outputMax = num
        if outputMax >= skippedMax: return outputMax
    
    return outputMax

def run():
    global arr, result
    arr = []
    for i in range(choiceCount): arr.append(i+1)
    result = [0] * choiceCount
    
    for trial in range(trialCount):
        if trial % feedBackSpace == 0 and printToggle: print(f"Computing trial {trial}/{trialCount}")
        random.shuffle(arr)
        
        for stopPoint in range(choiceCount):
            result[stopPoint] = round(result[stopPoint] + pick(stopPoint, arr)/trialCount, precision)

def saveData():
    f = open('data.txt', 'a')
    f.write(f'{result.index(max(result))},') # Use index to save the optimal stopPoint
    f.close()

def deleteData():
    f = open('data.txt', 'w')
    f.write('')
    f.close()

def loadData():
    global dataString
    f = open('data.txt', 'r')
    dataString = f.read()
    f.close()

def plot():
    print(f"The best stopPoint is at {result.index(max(result))}/{choiceCount} with {max(result)}")
    arr.sort()
    xPoints = np.array(arr)
    yPoints = np.array(result)
    
    plt.plot(xPoints, yPoints)
    plt.show()

# Global variables
trialCount = 100
choiceCount = 157
precision = 4
repetition = 1000
feedBackSpace = 50
delay = 10
printToggle = False
dataString = ""
dataArr = []
arr = []
result = []

# Restart data
#deleteData()

# Main program
for repeat in range(repetition):
    if repeat % delay == 0: print(f"Computing repetition {repeat}/{repetition}")
    run()
    saveData()

loadData()
dataArr = dataString.split(',')
dataArr.pop()
dataArr = list(map(lambda s: int(s), dataArr))
average = sum(dataArr) / len(dataArr)

print(f"The optimal stop point is at index {round(average, 2)}/{choiceCount} ({round(average/choiceCount*100, 2)}%)")















