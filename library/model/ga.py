from lib2to3.pgen2.tokenize import generate_tokens
from typing import List
import numpy as np

from library.EA import Individual, Population

from .abstract_model import AbstractModel
from ..utils.operator import crossover, mutation, selection
from ..utils.load_data import Load

class model(AbstractModel):
    def compile(self, 
        data_loc: str, 
        crossover: crossover.OX_Crossover, mutation: mutation.SwapMutation, selection: selection.ElitismSelection):
        return super().compile(data_loc, crossover, mutation, selection)

    def fit(self, num_generations: int, num_individuals: int, prob_crossover= 0.7, prob_mutation = 0.3, *args, **kwargs):
        super().fit(*args, **kwargs)

        self.res = []
        population = Population(
            num_individual=num_individuals,
            dim=self.dim,
            data=self.data,
            seed=self.seed,
            eval_initial=True
        )

        for epoch in range(num_generations):
            # Initiate offspring population
            offsprings = Population(
                num_individual=0,
                dim=self.dim,
                data=self.data
            )

            while len(offsprings) < len(population):
                p1, p2 = population.__getRandomInds__(size=2)
                generate_new_individual = False
                if np.random.rand() < prob_crossover:
                    o1, o2 = self.crossover(parent1=p1, parent2=p2)
                    generate_new_individual = True
                elif np.random.rand() < prob_mutation:
                    o1 = self.mutation(p1)
                    o2 = self.mutation(p2)
                    generate_new_individual = True
                
                if generate_new_individual:
                    offsprings.__addIndividual__(o1)
                    offsprings.__addIndividual__(o2)
            
            # Merge and update rank
            # for i in offsprings.list_indiv:
            #     print(len(i))

            population = population + offsprings
            population.update_rank()
            best_indiv = population.get_best()
            self.res.append(best_indiv.fcost)
            print(f"Epoch {epoch+1}:\n{best_indiv.solution} - {best_indiv.fcost}")
            
            population = self.selection(population, num_individuals)
        return best_indiv
        

