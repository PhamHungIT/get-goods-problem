import numpy as np
import time

from library.utils.load_data import Load
from argparse import ArgumentParser

if __name__ == "__main__":

    parser = ArgumentParser(add_help=False)
    parser.add_argument("--dataset", type=str)
    param = parser.parse_args()
    
    data = Load()
    data(param.dataset)
    # data("library/data/txt_data/case_5.txt")

    N = data.num_kinds
    M = data.num_bins
    Q = data.mat_info
    d = data.mat_dis
    q = data.order
    X = np.full(M+1, -1)
    cost = 0
    min_cost = np.inf
    solution = np.copy(X)
    update_time = 1
    history_cost = []
    def Try(k):
        global Q, d, q, N, M, X, cost, min_cost, solution, update_time
        for v in range(1, M+1):
            if v not in X[:k]:
                # update the state
                X[k] = v
                cost += d[X[k-1], X[k]]
                q -= Q[:, v-1]

                if (cost + np.min(d) <= min_cost):
                    if np.all(q <= 0) or k == M:
                        if cost + d[X[k],0] < min_cost:
                            min_cost = cost + d[X[k],0]
                            solution = np.copy(X[:k+1])
                            print(f"Update {update_time}: \nPath: {solution} - Cost: {min_cost} \n")
                            update_time+=1
                            history_cost.append(min_cost)
                    else:
                        Try(k+1)
                # recover the state
                cost -= d[X[k-1], X[k]]
                q += Q[:, v-1]


    # Run
    X[0] = 0
    time_begin = time.time()
    Try(1)
    print(f"Time taken: {round(time.time() - time_begin, 2)}")
