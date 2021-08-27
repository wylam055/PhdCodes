from GraphClass import CausalGraph
import random
import numpy as np
import networkx as nx


##############################################################################################################

def SEM(coef_mat, sample_size, varLow=1, varHigh=3):
    coef_mat = np.array(coef_mat)
    assert coef_mat.shape[0] == coef_mat.shape[1]
    assert not np.any(np.diag(coef_mat))
    no_of_var = len(coef_mat)
    add_edges = []

    parent_dict = {}
    for i in range(no_of_var):
        parent_dict[i] = list(np.where(np.array(coef_mat[i]) != 0)[0])
        for parent in parent_dict[i]:
            add_edges.append((parent, i))

    nx_graph = nx.DiGraph()
    nx_graph.add_nodes_from(range(no_of_var))
    nx_graph.add_edges_from(add_edges, color='b')
    assert nx.is_directed_acyclic_graph(nx_graph)

    model = np.full((no_of_var, sample_size), np.float())
    cg = CausalGraph(no_of_var)

    done = []
    while True:
        for i in range(no_of_var):
            if i in done:
                continue

            elif len(parent_dict[i]) == 0 or set(parent_dict[i]).issubset(set(done)):
                var_e = random.uniform(varLow, varHigh)
                Xi = np.random.normal(0, var_e, sample_size)
                for parent in parent_dict[i]:
                    Xi += coef_mat[i][parent] * model[parent]
                    cg.addDirectedEdge(parent, i)
                model[i] = Xi
                done.append(i)

        if len(done) == no_of_var:
            break

    cg.data = model.transpose()
    cg.coef_mat = coef_mat
    cg.nx_graph = nx_graph
    cg.toNxSkeleton()
    cg.sample_size = sample_size
    return cg


##############################################################################################################

##############################################################################################################
### Example ##################################################################################################
##############################################################################################################

if __name__ == "__main__":
    ##########################
    ### Simulation setting ###
    ##########################
    coef_mat = [[0, 0.7, 0, 1.1],
                [0, 0, 0, 0],
                [0, 0, 0, -0.8],
                [0, 0, 0, 0]]
    sample_size = 100000
    cg = SEM(coef_mat, 1000000)
    print("Adjacency matrix")
    print("(where [x,y]=1 & [y,x]=0 represent x→y; [x,y]=[y,x]=0 represent x-y); non-adjacency as -1)")
    print(cg.adjmat, "\n")
    print("List of directed edges (where (x,y) represents x→y):\n", cg.findFullyDirected(), "\n")
    print("Coefficient matrix:\n", cg.coef_mat, "\n")
    BIC = cg.getBIC()
    print(f"BIC score: {BIC}")
    # np.savetxt("temp/test_data.csv", cg.data, delimiter=",")

##############################################################################################################
