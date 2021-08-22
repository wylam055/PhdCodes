from SEMSimulator import randomSEM
import numpy as np
import Helper

no_of_nodes = 4
avg_deg = 2
sample_size = 1000000
BIC_dict = Helper.createBICDict(no_of_nodes)

cg = randomSEM(no_of_nodes=no_of_nodes, avg_deg=avg_deg, coefLow=0.2, coefHigh=0.7, sample_size=sample_size,
               coefSymmetric = True, randomizeOrder=True)
cg.cov_mat = np.cov(cg.data, rowvar=False)
cg.sample_size = cg.data.shape[0]

print(Helper.BIC_graph(cg.adjmat, BIC_dict, cg.cov_mat, cg.sample_size, penalty=1))
print(cg.findFullyDirected())
print(cg.coef_mat)
#######################################################################################################################

np.savetxt("temp/test_data_" + str(no_of_nodes) + "_" + str(avg_deg) +"_" + str(sample_size) +
           ".csv", cg.data, delimiter=",")