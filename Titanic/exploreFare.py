import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BINS = 20

# Load data
trainData = pd.read_csv('train.csv')
survivedVector = trainData['Survived']
fareVector = trainData['Fare']

# Plot data
mostExpensiveTicket = fareVector.max()
cheapestTicket = fareVector.min()
bandwidth = (mostExpensiveTicket - cheapestTicket) / BINS

numOfSurvivedPeople = np.zeros(BINS + 1)
totalNumOfPeople = np.zeros(BINS + 1)

for i in range(len(fareVector)):
    binIndex = int(fareVector[i] / bandwidth)
    if survivedVector[i] == 1:
        numOfSurvivedPeople[binIndex] += 1
    totalNumOfPeople[binIndex] += 1

 # patch to avoid division by 0
for i in range(BINS + 1):
    if totalNumOfPeople[i] == 0:
        totalNumOfPeople[i] = 1

survivedProportion = numOfSurvivedPeople / totalNumOfPeople
plt.plot(survivedProportion)

plt.show()
