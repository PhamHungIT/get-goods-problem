from library.EA import *
from library.utils.operator.selection import ElitismSelection
from library.utils.operator.crossover import OX_Crossover
from library.utils.operator.mutation import SwapMutation
from library.model.ga import model
from library.utils.load_data import Load

from IPython.display import display
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import time


if __name__ == "__main__":

    parser = ArgumentParser(add_help=False)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--num_epoch", default=100, type=int)
    parser.add_argument("--pop_size", default=1000, type=int)
    parser.add_argument("--prob_m", default=0.2, type=float)
    parser.add_argument("--prob_c", default=0.7, type=float)
    parser.add_argument("--num_run", default=1, type=int )


    param = parser.parse_args()
    print("INFO - DATA:")
    data = Load()
    data(param.dataset)
    print(data)
    print(f"\nGENETIC ALGORITHM: \n\t+ Population: {param.pop_size} individuals\n\t+ Number generations: {param.num_epoch} \
        \n\t+ Probability crossover: {param.prob_c}\n\t+ Probability mutation: {param.prob_m}\n")

    sum_cost = 0
    sum_time = 0
    print("LOADING...")
    for num_run in range(param.num_run):
        time_begin = time.time()
        print(f"Run {num_run + 1}/{param.num_run}: ")
        ga_model = model()

        ga_model.compile(
            data_loc=param.dataset,
            crossover=OX_Crossover(),
            mutation=SwapMutation(),
            selection=ElitismSelection()
        )
        solution = ga_model.fit(
        num_generations=param.num_epoch,
        num_individuals=param.pop_size,
        prob_crossover=param.prob_c,
        prob_mutation=param.prob_m
        )
        print(f"\nComplete {num_run+1}/{param.num_run}")
        print(solution)

        sum_time += round(time.time() - time_begin, 2)
        sum_cost += solution.fcost
        if (num_run == 0):
            best_solution = solution
            worst_solution = solution
        else:
            if (solution.fcost < best_solution.fcost):
                best_solution = solution
            elif (solution.fcost > worst_solution.fcost):
                worst_solution = solution


    print("-"*100)
    print(f"\nResult GA - {param.num_epoch} epoch - {param.pop_size} individuals, after {param.num_run} times run:")
    print(f"  - Best solution:\n {best_solution}")
    print(f"  - Worst solution:\n {worst_solution}")
    print(f"  - Average of cost: {sum_cost/param.num_run}")
    seconds = sum_time/param.num_run

    minutes = seconds // 60 
    seconds = seconds - minutes * 60 
    display("  - Average of time: %02dm %2.02fs "%(minutes, seconds))

    # plt.plot(ga_model.res)
    # plt.title("Convergence process")
    # plt.xlabel("Epoch")
    # plt.ylabel("Cost")
    # plt.show()