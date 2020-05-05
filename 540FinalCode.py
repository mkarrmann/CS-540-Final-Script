import numpy as np
from sklearn.linear_model import LassoCV
import scipy
import csv
import matplotlib.pyplot as plt
import math

def load_data(filepath):
    """Data should be in format of rows corresponding to location, and columns
    corresponding to Population, Population Density, Mean Annual Temperature,
    Passenger-Kilometres of Rail Transport per Year, Global Health Security Index,
    Average Social Distancing Score, p, alpha, beta"""
    reader = csv.reader(open(filepath, "rb"), delimiter=",")
    return np.array(list(reader)).astype("float")

def transformRow(row):
    """Transform row of data"""
    
    arr = []
    for i in range(len(row)):
        for j in range(i, len(row)):
            arr.append(row[i] * row[j])
    
    for k in range(len(row)):
        append(row[i])
        
    arr.append(1)
            
    return np.array(arr)

def transformDataMatrix(matrix):
    """Transform data matrix"""
    # Transormed row is a dimension (n + d) choose n
    newMatrix = np.zeros(np.size(matrix, 0), scipy.special.binom(np.size(matrix, 0) + 2, np.size(matrix, 0)))
    for i in range(np.size(matrix, 0)):
        newMatrix[i,:] = transformRow(matrix[i,:])
        
    return newMatrix

def fitData(Phi, y):
    """Returns trained weights"""
    reg = LassoCV().fit(Phi, y)
    return reg.coef_

def getWeights(filepath):
    """Gets all three trained weights"""
    data = load_data(filepath)
    inputParams = data[:,:-3] # Gets input data
    Phi = transformDataMatrix(inputParams) # Transforms data
    yP = data[:,-3]
    yAlpha = data[:,-2]
    yBeta = data[:,-1]
    
    return fitData(Phi, yP), fitData(Phi, yAlpha),fitData(Phi, yBeta)

def graphPrediction(p, alpha, beta):
    """Graphs projected curve of deaths due to COVID-19 given parameters"""
    x = np.linspace(-3, 3)
    func = p / 2 * ((1 + (2 / math.sqrt(math.pi) * scipy.special.erf(alpha * (x - beta)))))
    plt.plot(x, scipy.special.erf(2*x))
    plt.xlabel('$x$')
    plt.ylabel('Total Deaths Due to COVID-19')
    plt.show()

if __name__ == '__main__':
    """Given made up test data, graphs several project curves of deaths due to
    COVID-19 under different social distancing mandates"""
    wP, wAlpha, wBeta = getWeights(filepath)
    testData = []
    points = np.array([50000000, 94, 60, 500, 80])
    for socDistScore in [0, 0.2, 0.5, 0.8, 1]:
        testData.append(transformRow(points + [socDistScore]))
        
    for data in testData:
        pHat = wP@data
        alphaHat = wAlpha@data
        betaHat = wBeta@data
        
        graphPrediction(pHat, alphaHat, betaHat)

        
    
