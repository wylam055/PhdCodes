#######################################################################################################################
from GraphClass import toPattern, CausalGraph
from itertools import permutations
from SEMSimulator import randomSEM
import numpy as np
from itertools import combinations
from Tester_Markov import CMCTester
import pandas as pd
from copy import deepcopy
#######################################################################################################################

#######################################################################################################################

def causalOrder(no_of_nodes):
    # Let m = no_of_nodes. This function returns m! permutations one by one.
    for order in permutations(range(no_of_nodes), no_of_nodes):
        yield order

#######################################################################################################################

def exhaustPermutations(cg, test_name, alpha):
    cg.setTestName(test_name)
    cg.corr_mat = np.corrcoef(cg.data, rowvar=False) if test_name == "Fisher_Z" else []

    permutationsDict = {}
    [CMC, I_G_star] = CMCTester(cg, test_name, alpha)

    if not CMC:
        return [[], 0, 0, []]

    nodes = cg.variables
    no_of_nodes = len(nodes)

    CI_facts = deepcopy(I_G_star)
    CD_facts = []

    def DAG_construct(pi):
        pi_edges = []
        for (j, k) in combinations(nodes, 2): # j < k guaranteed
            pi_j = pi[j]
            pi_k = pi[k]
            pi_cond_set = tuple(sorted(pi[0:j] + pi[j+1:k]))
            if (pi_j, pi_k, pi_cond_set) in CI_facts:
                continue
            elif (pi_j, pi_k, pi_cond_set) in CD_facts:
                pi_edges.append((pi_j, pi_k))
                continue
            else:
                p = cg.ci_test(pi_j, pi_k, pi_cond_set)
                if p > alpha:
                    CI_facts.append([pi_j, pi_k, pi_cond_set])
                    CI_facts.append([pi_k, pi_j, pi_cond_set])
                    continue
                else:
                    CD_facts.append([pi_j, pi_k, pi_cond_set])
                    CD_facts.append([pi_k, pi_j, pi_cond_set])
                    pi_edges.append((pi_j, pi_k))
                    continue
        return pi_edges

    pattern_list = []
    pi_directory = []

    for pi in causalOrder(no_of_nodes):
        pi_edges = DAG_construct(pi)
        pi_cg = CausalGraph(no_of_nodes)
        for (j, k) in pi_edges:
            pi_cg.addDirectedEdge(j, k)
        pattern = toPattern(pi_cg, checkDAG=False)
        existing = False
        for i in range(len(pattern_list)):
            if np.equal(pattern_list[i], pattern.adjmat).all():
                existing = True
                pi_directory.append([pi, i, len(pi_edges)])
        if not existing:
            pattern_list.append(pattern.adjmat)
            pi_directory.append([pi, len(pattern_list)-1, len(pi_edges)])

    results = pd.DataFrame(pi_directory, columns = ['Permutation', "MEC index", "No. of edges"])
    number_of_MECs = max(results.iloc[:, 1]) + 1
    fewest_number_of_edges = min(results.iloc[:, 2])
    FEC = results[results['No. of edges'] == fewest_number_of_edges]
    return [results, number_of_MECs, fewest_number_of_edges, FEC]

#######################################################################################################################

if __name__ == "__main__":
    ##########################
    ### Simulation setting ###
    ##########################
    no_of_nodes = 5
    avg_deg = 2
    coefLow = 0.2
    coefHigh = 0.7
    sample_size = 1000
    alpha = 0.01
    testName = "Fisher_Z"
    cg = randomSEM(no_of_nodes=no_of_nodes, avg_deg=avg_deg, coefLow=coefLow, coefHigh=coefHigh,
                   sample_size=sample_size, coefSymmetric = True, randomizeOrder = True)
    [results, number_of_MECs, fewest_number_of_edges, FEC] = exhaustPermutations(cg, testName, alpha)
    print("All SGS-minimal DAGs:")
    print(results, "\n")
    print(f"Number of MECs: {number_of_MECs}")
    print(f"Fewest number of edges: {fewest_number_of_edges} \n")
    print("All DAGs in the Frugal Equivalence Class:")
    print(FEC)

#######################################################################################################################
