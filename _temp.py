from SEMSimulator import randomSEM
from GraphClass import CausalGraph
import numpy as np
import Helper
from numpy import log as ln

no_of_nodes = 4
BIC_dict = Helper.createBICDict(no_of_nodes)

cg = randomSEM(no_of_nodes=no_of_nodes, avg_deg=2, coefLow=0.2, coefHigh=0.7, sample_size=1000000,
               coefSymmetric = True, randomizeOrder=True)
cg.cov_mat = np.cov(cg.data, rowvar=False)
cg.sample_size = cg.data.shape[0]

print(Helper.BIC_graph(cg.adjmat, BIC_dict, cg.cov_mat, cg.sample_size, penalty=1))
print(cg.findFullyDirected())
print(cg.coef_mat)
#######################################################################################################################

np.savetxt("temp/test_return_data.csv", cg.data, delimiter=",")