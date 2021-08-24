#######################################################################################################################
from GraphClass import toPattern, CausalGraph
from Tester_Markov import CMCTester
from itertools import permutations
import numpy as np
import networkx as nx
from itertools import combinations
from SEMSimulator import randomSEM
from copy import deepcopy
#######################################################################################################################

#######################################################################################################################

def causalOrder(no_of_nodes):
    # Let m = no_of_nodes. This function returns m! permutations one by one.
    for order in permutations(range(no_of_nodes), no_of_nodes):
        yield order

#######################################################################################################################

def permutationBasedRazorsTester(cg, test_name, alpha, **kwargs):
    """Test Pearl Minimality, frugality, and unique frugality
    :param cg: CausalGraph object
    :param test_name: name of the independence test being used (string)
    :param alpha: desired significance level in (0, 1) (float)
    :return:
    1. SGS: True if SGS-minimality is satisfied, and False otherwise
    2. Pm: True if P-minimality is satisfied, and False otherwise
    3. Fr: True if frugality is satisfied, and False otherwise
    4. uFr: True if u-frugality is satisfied, and False otherwise
    5. CI_facts: a list of conditional independence relations
    6. CD_facts: a list of conditional dependence relations
    """
    cg.setTestName(test_name)
    cg.getCorrMatrix() if test_name == "Fisher_Z" else []

    if "CMC_result" in kwargs:
        CMC = kwargs["CMC_result"][0]
        I_G_star = kwargs["CMC_result"][1]
    else:
        [CMC, I_G_star] = CMCTester(cg, test_name, alpha)

    if not CMC:
        return [False, False, False, False, [], []]

    CI_facts = kwargs["CI_facts"] + I_G_star if "CI_facts" in kwargs else deepcopy(I_G_star)
    CD_facts = kwargs["CD_facts"] if "CD_facts" in kwargs else []

    nodes = cg.variables
    no_of_nodes = len(nodes)
    no_of_true_edges = len(cg.findFullyDirected())
    true_adj = set([(i, j) for (i, j) in cg.findAdj() if i < j])
    true_directed = set(cg.findFullyDirected())

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

    SGS = True
    P_minimal = True
    Frugal = True
    U_frugal = True

    true_pattern = toPattern(cg, checkDAG=False)
    true_pattern_adjmat = true_pattern.adjmat

    for pi in causalOrder(no_of_nodes):
        pi_edges = DAG_construct(pi) # pi_edges constitute a SGS-minimal DAG G_pi.

        # Case 1: When |E(G_pi)| > |E(G*)|, G_pi cannot prove the non-P-minimality or non-frugality of G*
        if len(pi_edges) > no_of_true_edges:
            continue

        # Case 2: When |E(G_pi)| = |E(G*)|, G_pi is possibly a maximally frugal DAG
        # We check if G_pi is in the same MEC with G*
        elif len(pi_edges) == no_of_true_edges:
            if Frugal and U_frugal:
                pi_cg = CausalGraph(no_of_nodes)
                for (j, k) in pi_edges:
                    pi_cg.addDirectedEdge(j, k)
                pi_pattern = toPattern(pi_cg, checkDAG=False)
                if np.equal(true_pattern_adjmat, pi_pattern.adjmat).all():
                    continue
                else:
                    U_frugal = False

        # Case 3: When |E(G_pi)| < |E(G*)|, frugality fails but we still need to know whether G* is P-minimal or SGS_minimal
        else:
            Frugal = False
            # If I(G*) is a proper subset of I(G_pi), adj(G_pi) is a proper subset of adj(G*)
            pi_adj = set([(i, j) for (i, j) in pi_edges if i < j] + [(j, i) for (i, j) in pi_edges if i > j])
            if not pi_adj.issubset(true_adj): # If subset, then proper subset (because of Case 3)
                continue # Look for next G_pi
            else:
                # If E(G_pi) is a proper subset of E(G*), SGS-minimality fails
                if set(pi_edges).issubset(true_directed): # If subset, then proper subset (because of Case 3)
                    SGS = False
                    break
                else:
                    # If SGS-minimality has not been violated, we check whether I(G*) is a subset of I(G_pi).
                    # We construct the nx_graph object for G_pi to check d-separation.
                    pi_nx_graph = nx.DiGraph()
                    pi_nx_graph.add_nodes_from(nodes)
                    pi_nx_graph.add_edges_from(pi_edges)
                    next_pi = False
                    for CI in I_G_star:
                        if not nx.d_separated(pi_nx_graph, {CI[0]}, {CI[1]}, CI[2]):
                            next_pi = True
                            break
                    if next_pi:
                        continue # Look for next pi
                    else:
                        P_minimal = False

    if not SGS:
        return [False, False, False, False, CI_facts, CD_facts]
    elif not P_minimal:
        return [True, False, False, False, CI_facts, CD_facts]
    elif not Frugal:
        return [True, True, False, False, CI_facts, CD_facts]
    elif not U_frugal:
        return [True, True, True, False, CI_facts, CD_facts]
    else:
        return [True, True, True, True, CI_facts, CD_facts]

#######################################################################################################################

if __name__ == "__main__":
    ##########################
    ### Simulation setting ###
    ##########################
    no_of_nodes = 4
    avg_deg = 2
    coefLow = 0.2
    coefHigh = 0.7
    sample_size = 1000
    number_of_runs = 50
    alpha = 0.01
    testName = "Fisher_Z"

    ######################
    ### Run simulation ###
    ######################
    CMC_sum  = 0
    SGS_sum = 0
    Pm_sum = 0
    Fr_sum = 0
    uFr_sum = 0
    for i in range(number_of_runs):
        cg = randomSEM(no_of_nodes=no_of_nodes, avg_deg=avg_deg, coefLow=coefLow, coefHigh=coefHigh,
                       sample_size=sample_size, coefSymmetric = True, randomizeOrder = True)
        CMC, I_G_star = CMCTester(cg, testName, alpha)
        [SGS, Pm, Fr, uFr, CI_facts, CD_facts] = permutationBasedRazorsTester(cg, testName, alpha, CMC_result = [CMC, I_G_star])
        CMC_sym = '\u2713' if CMC else 'x'
        SGS_sym = '\u2713' if SGS else 'x'
        Pm_sym = '\u2713' if Pm else 'x'
        Fr_sym = '\u2713' if Fr else 'x'
        uFr_sym = '\u2713' if uFr else 'x'
        CMC_sum += CMC
        SGS_sum += SGS
        Pm_sum += Pm
        Fr_sum += Fr
        uFr_sum += uFr
        print(f"Run {i+1}: CMC {CMC_sym}, SGS {SGS_sym}, Pm {Pm_sym}, Fr {Fr_sym}, uFr {uFr_sym}")
    print("\n")
    print(f"CMC is satisfied in {CMC_sum} out of {number_of_runs} runs.")
    print(f"u-frugality is satisfied in {uFr_sum} out of {number_of_runs} runs.")
    print(f"frugality is satisfied in {Fr_sum} out of {number_of_runs} runs.")
    print(f"P-minimality is satisfied in {Pm_sum} out of {number_of_runs} runs.")
    print(f"SGS-minimality is satisfied in {SGS_sum} out of {number_of_runs} runs.")

#######################################################################################################################