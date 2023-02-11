'''
Hybridizing cuckoo search algorithm with biogeography-based optimization

Author: aluizgc
'''

import numpy as np
from scipy.special import gamma

# Parameter settings for the SCBH
NP = 100  # population size
pa = 0.3  # discovery probability
alpha = 1.1  # alpha> 0  is a parameter determining the step size of cuckooo
beta = 1.7  # is Levy flight exponent
delta = 1.6
I = 1.   # maximum imigration
E = 1.   # imigration rate
maxFES = 2000
runs = 10

lbound = np.array([0.0, 0.0, 0.0, 1.0, 0.0])  # I_ph I_0 R_s n R_sh
ubound = np.array([1.0, 1e-6, 0.5, 2.0, 100.0])
dimension = len(lbound)

# Compute migration rates, assuming the population is sorted from most fit to least fit
mu = E*((NP-(np.arange(NP)+1.))/NP)  # emigration rate
lambdaa = I*(1. - mu)  # immigration rate

# --- Define constants and import data --- #
charge = 1.6021766e-19
boltz = 1.38065e-23
temperature = 33 + 273.15  # in Kelvin

# --- Define objective function (RMSE) --- #
def objective(indv):
    simulated = indv[0]-indv[1]*(np.exp(charge*(voltage+current*indv[2])/(
        indv[3]*boltz*temperature), dtype=np.float64)-1)-(voltage+current*indv[2])/indv[4]
    return np.power((1/np.float(np.size(voltage)))*np.sum(np.power(current-simulated, 2, dtype=np.float64), dtype=np.float64), 0.5, dtype=np.float64)

# --- Penalty function --- #
def Penalty(sol):
    for j in range(dimension):
        if sol[j] > ubound[j]:
            sol[j] = lbound[j] + np.random.rand()*(ubound[j]-lbound[j])
        if sol[j] < lbound[j]:
            sol[j] = lbound[j] + np.random.rand()*(ubound[j]-lbound[j])
    return sol
results = np.zeros((runs,dimension))
costs = np.zeros((runs,40))
for run in range(runs):
    data = np.loadtxt('curva_0.txt')
    voltage = data[:, 0]
    current = data[:, 1]
    
    # Random initial solutions
    nest = lbound+np.random.rand(NP, dimension)*(ubound-lbound)
    fit_nest = np.zeros(NP)
    for i in range(0, NP):
        fit_nest[i] = objective(nest[i, :])
    fmin, index = np.min(fit_nest), np.argmin(fit_nest)
    bNest = nest[index, :]

    nest_new = nest.copy()
    fit_nest_new = fit_nest.copy()

    for FES in range(1,maxFES+1):
        sigma_u = (gamma(1.+beta)*np.sin(np.pi*beta/2.) /
                (gamma((1.+beta)/2.)*beta*2.**((beta-1.)/2.)))**(1./beta)
        sigma_v = 1.
        nest_mean = np.mean(nest)
        for i in range(0, NP):
            u = np.random.randn(dimension)*sigma_u
            v = np.random.randn(dimension)*sigma_v
            sr = np.random.rand()
            if (sr > 2./3.):
                s = nest[i, :]+alpha * \
                    ((0.01*u)/(np.abs(v)**(1./beta)))*(nest[i, :] - bNest[:])
            elif ((sr <= 2./3.) and (sr > 1./3.)):
                s = nest_mean + delta * \
                    np.log(1./np.random.rand())*(nest_mean - nest[i, :])
            else:
                s = nest[i, :]+delta*np.exp(np.random.rand())*(bNest-nest[i, :])
            nest_new[i, :] = Penalty(s)
            fit_nest_new[i] = objective(nest_new[i, :])
    # if better than
        for i in range(0, NP):
            if (fit_nest_new[i] < fit_nest[i]):
                fit_nest[i] = fit_nest_new[i]
                nest[i, :] = nest_new[i, :]
        fmin, index = np.min(fit_nest), np.argmin(fit_nest)
        bNest = nest[index, :]
    
    # Biogeography-based discovery operator
        sort = np.argsort(fit_nest)
        fit_nest = fit_nest[sort]
        nest = nest[sort]
        for i in range(0, NP):
            for j in range(0, dimension):
                if (np.random.rand() > pa):
                    pp = mu/np.sum(mu)
                    cum_pp = np.cumsum(pp)
                    SelectIndex = np.where(np.random.rand() < cum_pp)
                    while (SelectIndex[0][0] == i):
                        SelectIndex = np.where(np.random.rand() < cum_pp)
                    Alpha = alpha  # np.random.rand()
                    
                    # migration step
                    nest_new[i, j] = Alpha*nest[i, j] + \
                        (1.-Alpha)*nest[SelectIndex[0][0], j]
                else:
                    nest_new[i, j] = nest[i, j]
        # Selection
        for i in range(0, NP):
            nest_new[i, :] = Penalty(nest_new[i, :])
            fit_nest_new[i] = objective(nest_new[i, :])
            if (fit_nest_new[i] < fit_nest[i]):
                fit_nest[i] = fit_nest_new[i]
                nest[i, :] = nest_new[i, :]
        fmin, index = np.min(fit_nest), np.argmin(fit_nest)
        bNest = nest[index, :]
        if FES%50 == 0:
            costs[run][int(FES/50)-1] = fit_nest[0]
        print('Run: ', run)
        print('It: ', FES)
        print('%.4E' %fit_nest[0], nest[0, :])
        
    results[run] = nest[0,:]
    np.savetxt('BHCSrun.txt', results, delimiter= ' ')
    np.savetxt('BHCScosts.txt',costs)    
