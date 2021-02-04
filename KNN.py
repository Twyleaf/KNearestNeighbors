import pandas as pd
from math import sqrt

def main():
    dataset = pd.read_csv('Dataset_CancerClassification/Dados_Normalizados/cancer_train.csv')
    
    test = pd.read_csv('Dataset_CancerClassification/Dados_Normalizados/cancer_test.csv')
    KNNEvaluate(1,dataset,test)

def euclideanDistance(row1, row2):
    distance = 0.0
    ##print (row2)
    for name, value in row1.iteritems():
        distance += (value - row2[name]) ** 2
    return sqrt(distance)

def KNNEvaluate(KValue,dataset,test):
    totalTests = 0
    correctCount = 0
    for testRowIndex, testRow in test.iterrows():
        KNearestNeighbors = getNearestNeighbors(KValue,testRow,dataset)
        testRowPredictedValue = getMedianValue(KNearestNeighbors, KValue)
        totalTests+=1

        if testRowPredictedValue == testRow['target']:
            correctCount+=1
            print("acertou")
        else: 
            print("errou")
    print("acertou " + str(correctCount) + " de " + str(totalTests))
    print("taxa de acerto de " + str(correctCount*100 / totalTests) + "%")

def getNearestNeighbors(KValue,targetRow,dataset):
    distances = list()
    for dataRowIndex, dataRow in dataset.iterrows():
        dist = euclideanDistance(targetRow.drop('target'), dataRow.drop('target'))
        distances.append((dataRow, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(KValue):
        neighbors.append(distances[i][0])
    return neighbors

def getMedianValue(nearNeighbors, neighborCount):
    neighborsTotal = 0
    for neighbor in nearNeighbors:
        neighborsTotal += neighbor['target']
    print (neighborCount/2)
    if neighborsTotal >= neighborCount/2:
        return 1.0
    else: return 0.0


main()
