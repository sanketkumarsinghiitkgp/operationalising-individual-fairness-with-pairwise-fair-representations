import numpy as np
import numpy.linalg as linalg

def formlaplacian(mat):
    csum = np.sum(mat,axis=0,dtype=float)
    d = np.diag(csum)
    return d-mat

#Optimisation def formlaplacian(mat): def formlaplacian(mat): # pre-optimization part, currently taking random matrices and vectors x = np.random.rand(100,1000) d = 50
gamma = 0.5
N = 100
D = 1000
wx = np.random.rand(N,N)
wf = np.random.rand(N,N)
##


#Optimisation

Lx=formlaplacian(wx)
Lf=formlaplacian(wf)
L=gamma*Lf+(1-gamma)*Lx
xt=np.transpose(x)
mat=np.dot(x,np.dot(l,xt))
eigenValues, eigenVectors = linalg.eig(mat)
dx = eigenValues.argsort()[::-1]   
eigenValues = eigenValues[idx]
eigenVectors = eigenVectors[:,idx]
V=eigenVectors[0:D]
z=np.dot(V.transpose(),x)

#Experiment/ Logistic regression after this
