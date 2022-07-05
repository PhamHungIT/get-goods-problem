import numpy as np

from ..utils.operator import crossover, mutation, selection
from ..utils.load_data import Load

class AbstractModel:
    def __init__(self, seed=None) -> None:
        self.seed = seed
        if seed is None:
            pass
        else:
            np.random.seed(seed)
    def compile(self, data_loc: str, crossover: crossover.AbstractCrossover, mutation: mutation.AbstractMutation, selection: selection.AbstractSelection):
        self.data = Load()
        self.data(data_loc)

        self.dim = self.data.num_bins + 1
        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection
    def fit(self, *args, **kwargs):
        pass

