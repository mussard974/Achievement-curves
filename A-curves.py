#############################################################
# Mussard & Pi Alperin 2020 : \alpha-\nu Achievement curves #
#############################################################

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import scipy.stats as ss
from sklearn.utils import resample



#############################################################
#                       Function                            #
#############################################################

def A_curves(X = float, alpha =  float, weight = float, method = True, curve = True, confidence_interval = None):
    
    """
    alpha = risk aversion parameter in [0,200]
    weight = "Betti Verma" or "Cerioli Zani" or "equal weight"    
    method = "risk neutral" or "risk aversion" or "extreme risk" 
    curve = "order 2" or "order 3" or "order 4" (value of \nu)
    confidence_interval = value of the confidence level in [0.01,0.1]
    """
    
    X = X[X[:,0].argsort()]             # sorting the ranks from the lowest to the highest
    n, k = X.shape
    x = X[:,1:k]                        # shortfall in [0,1] by individuals over k-1 dimensions #
    
    rank_diff, rank = np.zeros((n,1)), np.zeros((n,1))
    rank = np.array(X[:,0])             # rank = percentiles of the equivalent income between 0 and 1 
    rank_diff[0] = rank[0] 
    for i in range(1,n-1):
        rank_diff[i] = rank[i+1]-rank[i] 
    
    # Weight vector #
    assert weight in ["Betti Verma", "Cerioli Zani", "equal weight"], "weight for each dimension must be 'Betti Verma', 'Cerioli Zani' or 'equal weight'" 
    w = np.zeros((1,k-1))                  
    if weight == "Betti Verma": 
        c = np.corrcoef(x.T)
        e=[]
        for i in range(k-1):
            for j in range(i+1,k-1):
                e.append(c[i,j])  
        A = np.zeros((len(e)-1, len(e)-1))
        for i in range(len(e)):
            for j in range(i+1,len(e)):
                A[i,j-1] = np.abs(e[i] - e[j])
        rho_H = np.max(A)               # rho_H cut-off 
        for i in range(k-1):
            b1, b2 = 0, 0
            for j in range(k-1):
                if j!=i:
                    if np.corrcoef(x[:,i], x[:,j])[0,1] >= rho_H:
                        b1 += np.corrcoef(x[:,i], x[:,j])[0,1]
                    else:
                        b2 += np.corrcoef(x[:,i], x[:,j])[0,1]
            w[:,i] = ss.variation(x[:,i])*(1/(1+b1))*(1/(1+b2))  
    if weight == "Cerioli Zani":    
        w = -np.log(np.mean(x, axis = 0))              
    if weight == "equal weight":       
            w = 1/(k-1)
    
    # achievement vector #
    assert alpha >= 0 and alpha <= 200, "Risk sensibility parameter alpha must be in [0,200]"
    a = np.zeros((n,1))                
    if method == "risk neutral":       # Type of risk
        for i in range(n):
            a[i] = np.prod(np.power(x[i,:].T, w.T))
            if np.all(x[i,0:k-2]==0) and x[i,k-2]==1:
                a[i] = 1     
    if method == "risk aversion":       # Type of risk
        for i in range(n):
            a[i] = ((w*np.power(x[i,:], alpha)).sum())**(1/alpha)
    if method == "extreme risk":        # Type of risk
        a = np.amax(x, axis=1)
    
    # Achievement CURVES : AC #  
    a = np.ones((n,1)) - a              # 1 - a = achievement 
    a = a/(rank_diff.T @ a)
    AC = np.zeros((n,3))                # Achievement Curves Order 2 & 3 & 4
    AC[0,0] = rank[0]*a[0]/2            
    AC[0,1] = rank[0]*AC[0,0]/2
    AC[0,2] = rank[0]*AC[0,1]/2
    for i in range(1,n):
        AC[i,0] = AC[i-1,0] + ((a[i-1] + a[i])*(rank[i]-rank[i-1]))/2            # order 2 
        AC[i,1] = AC[i-1,1] + ((AC[i-1,0] + AC[i,0])*(rank[i]-rank[i-1]))/2      # order 3
        AC[i,2] = AC[i-1,2] + ((AC[i-1,1] + AC[i,1])*(rank[i]-rank[i-1]))/2      # order 4
          
    # Bootstrap Confidence Interval : CI #
    if confidence_interval != None:
        assert confidence_interval >= 0.01 and confidence_interval <= 0.1, "level for confidence_interval must be in [0.01,0.1]"
        B = 5000
        for j in range(2):
            CI = np.zeros((n,2))
            Bootstrap = np.zeros((n,B))
            for i in range(B):
                Bootstrap[:,i] = np.sort(resample(AC[:,j]), axis=0) 
            for i in range(n):
                Bootstrap[i,:] = np.sort(Bootstrap[i,:])
                CI[i,0], CI[i,1] = Bootstrap[i,int(B*confidence_interval/2)], Bootstrap[i,int(B*(1-confidence_interval/2))]   # order2 
            if curve == "order 2" and j== 0:
                return CI[:,0], AC[:,j], CI[:,1]
            if curve == "order 3" and j==1:
                return CI[:,0], AC[:,j], CI[:,1]
            if curve == "order 4" and j==2:
                return CI[:,0], AC[:,j], CI[:,1]
    else:
        if curve == "order 2":
            return AC[:,0]
        if curve == "order 3":
            return AC[:,1]       
        if curve == "order 4":
            return AC[:,2]       
    

    
#############################################################
#                   Application on Data                     #
#############################################################

## DATA smoking ##
data = "data_risk_smoke.txt"
X = np.asarray(genfromtxt(data))

## DATA alcohol ##
data1 = "data_risk_alcohol.txt"
XX = np.asarray(genfromtxt(data1))

## PLOT : comparing two curves - order 2 ##
curve1 = A_curves(X, alpha = 100, weight="Betti Verma", method="risk aversion", curve="order 4", confidence_interval = None)
curve2 = A_curves(XX, alpha = 100, weight="Betti Verma", method="risk aversion", curve="order 4", confidence_interval = None)
plt.plot(X[:,0], curve1, 'r', label='A-curve_smoking (order 4)')
plt.plot(X[:,0], curve2, 'g', label='A-curve_alcohol (order 4)')
plt.ylabel('Cumulated achievement')
plt.xlabel('Income Percentiles')
plt.title('Achievement Curves')
plt.legend(loc='best')
plt.show()

## PLOT curve1 with confidence intervals - order 3 - level = 0.01 ##
curve1 = A_curves(X, alpha = 100, weight="equal weight", method="risk aversion", curve="order 3", confidence_interval = 0.01)
plt.plot(X[:,0], curve1[0], 'b', label='CI_inf')
plt.plot(X[:,0], curve1[1], 'r', label='A-curve_smoking (order 3)')
plt.plot(X[:,0], curve1[2], 'b', label='CI_sup')
plt.ylabel('Cumulated achievement')
plt.xlabel('Income Percentiles')
plt.title('Achievement Curves')
plt.legend(loc='best')
plt.show()