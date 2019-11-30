'''
Modified Artificial Bee Colony Algorithm

Author: aluizgc
'''

import numpy as np
from scipy.constants import e, Boltzmann

np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(precision=4)

#-- Variables --#

qE = e  # electron charge (C)
boltzmann = Boltzmann  # Bolztmann constant (J/K)
temperature = 33. + 273.15
maxIter = 20000
currIter = 0
lbound = np.array([0.0, 0.0, 0.0, 0.0, 1.0]) #SDM
ubound = np.array([1.0, 1e-6, 0.5, 100.0, 2.0]) #SDM
# bound = [Iph (A), Isd (muA), Rs (ohm), Rsh (ohm), n (diode ideality factor) ] SDM

wmax = 1.
wmin = 0.7
m = len(lbound)  # number of variables
SN = 100  # populaton size
limit = 0.06*maxIter  # abandonment limit
runs = 10
results = np.zeros((runs,m))
costs = np.zeros((runs,40))



#-- Cost function (SDM)--#

def costFunc(x):
    ISim = x[0]-x[1]*(np.exp(qE*(voltage+current*x[2])/(
        x[4]*boltzmann*temperature), dtype=np.float64)-1)-(voltage+current*x[2])/x[3]
    return np.power((1/np.float(np.size(voltage)))*np.sum(np.power(ISim-current, 2, dtype=np.float64), dtype=np.float64), 0.5, dtype=np.float64)

#-- Fitness function --#

def fitness(foodSource):
    costF = np.zeros((SN, 1))
    fit = np.zeros((SN, 1))
    for i in range(SN):
        costF[i] = costFunc(foodSource[i, :])
        if costF[i] >= 0:
            fit[i] = 1./(1.+costF[i])
        if costF[i] < 0:
            fit[i] = 1.+np.abs(costF[i])
    return fit

#-- Probability and Roulette Wheel functions --#

def prob(fit, foodSource):
    r = np.random.uniform(0., 1.)
    pn = np.zeros(SN)
    chosen = np.zeros((SN, m))
    for i in range(SN):
        pn[i] = fit[i]/np.sum(fit)
        if r < pn[i]:
            chosen[i] = foodSource[i,:]
    return chosen

#-- Penalty Function --#
def Penalty(sol):
    for j in range(m):
        if sol[j] > ubound[j]:
            sol[j] = lbound[j]+np.random.random()*(ubound[j]-lbound[j])
        if sol[j] < lbound[j]:
            sol[j] = lbound[j]+np.random.random()*(ubound[j]-lbound[j])
    return sol
    
###--- MABC MAIN LOOP ---###

#-- Initialization phase --#
# Initialize random food source
for run in range(runs):
    data = np.loadtxt('curva0.txt')
    voltage = data[:, 0]
    current = data[:, 1]
    Xn = np.zeros((SN, m))  # initial solution matrix
    Vn = np.zeros((SN, m))  # new solutions matrix
    bas = np.zeros(SN)  # abandonment counter
    k = 0
    costXn = np.zeros(SN)
    costVn = np.zeros(SN)
    olbXn = np.zeros((SN,m))
    olbVn = np.zeros((SN,m))
    costOLXn = np.zeros(SN)
    costOLVn = np.zeros(SN)
    olbXnIndex = np.zeros(SN)
    bestCost = 0
    index = 0
    bestSolution = np.zeros((1,m))
    
    for i in range(SN):
        for j in range(m):
            Xn[i][j] = lbound[j]+np.random.random()*(ubound[j]-lbound[j])
        costXn[i] = costFunc(Xn[i,:])
    index = np.argmin(costXn)
    bestSolution = Xn[index,:]
        
    for currIter in range(1,maxIter+1):
        SF = wmax - ((wmax-wmin)/currIter)*maxIter
    #-- Employed Bees Phase --#
    # Initialize a second food source based on the first one.
    # Do greedy selection and fitness calculation.
        for i in range(SN):
            for j in range(m):
                while (k == i):
                    k = np.random.randint(SN)
                Vn[i][j] = Xn[i][j] + np.random.uniform(-SF,SF)*(Xn[i][j]-Xn[k][j])
                k = 0
            Vn[i,:] = Penalty(Vn[i,:])
            costVn[i] = costFunc(Vn[i,:])
            if costVn[i] < costXn[i]:
                Xn[i,:] = Vn[i,:]
                bas[i] = 0
            else:
                bas[i] = bas[i]+1
            Xn[i,:] = Penalty(Xn[i,:])
            costXn[i] = costFunc(Xn[i,:])        

        fit = fitness(Xn)
        
        #-- Onlooker Bees Phase --#
        # Probabilities and roullete selection
        # Food source for onlooker bees (Xn)
        while not olbXn.any():
            olbXn = prob(fit, Xn)

        # Not zeros index for olbXn
        for i in range(SN):
            if np.all(olbXn[i,:] != 0):
                olbXnIndex[i] = i
        for t in range(SN):
            for j in range(m):
                while (k == olbXnIndex[t]):
                    k = np.random.randint(SN)
                if t == olbXnIndex[t]:
                    if olbXnIndex[t] == 0:
                        olbVn[t,:] = olbXn[t,:]
                    else:
                        olbVn[t][j] = olbXn[t][j] + np.random.uniform(-SF,SF) * (olbXn[t][j]-olbXn[k][j])
                k = 0
            costOLXn[t] = costFunc(olbXn[t,:])
            costOLVn[t] = costFunc(olbVn[t,:])
            if  costOLVn[t] < costOLXn[t]:
                olbXn[t,:] = olbVn[t,:]
            olbXn[t,:] = Penalty(olbXn[t,:])
        
        # Comparision
        for i in range(SN):
            if costOLXn[i] < costXn[i]:
                Xn[i,:] = olbXn[i,:]
                bas[i] = 0
            else:
                bas[i] = bas[i]+1
            Xn[i,:] = Penalty(Xn[i,:])
        bestCost = costFunc(bestSolution)
        #-- Update best solution --#
        for i in range(SN):
            if costXn[i] <= bestCost:
                bestSolution = Xn[i,:]
                bas[i] = 0
        bestCost = costFunc(bestSolution)
        
        #-- Scout bees phase --#
        for i in range(SN):
            if bas[i] >= limit:
                for j in range(m):
                    Xn[i][j] = bestSolution[j]*(1+np.random.rand())
                    bas[i] = 0
            Xn[i,:] = Penalty(Xn[i,:])
        
        if currIter%500 == 0 :
            costs[run][int(currIter/500)-1] = bestCost
    
        #-- Print result for each iteration
        print('Run: ', run)
        print('It: ',currIter)
        print('%.4E' %bestCost, bestSolution)
    #results[run] = bestSolution
    np.savetxt('MABCrun{}.txt'.format(run), bestSolution, delimiter= ' ')
    np.savetxt('MABCcosts{}.txt'.format(run), costs)    


