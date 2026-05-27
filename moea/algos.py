import math
import random
import functools
import moea.ema_patch
from platypus import AbstractGeneticAlgorithm
from platypus._math import POSITIVE_INFINITY
from platypus.operators import RandomGenerator, TournamentSelector
from platypus.core import (nondominated_sort, nondominated_truncate, HypervolumeFitnessEvaluator,
                           AttributeDominance, fitness_key, Direction, EpsilonBoxArchive)
from platypus._tools import only_keys_for, remove_keys
from platypus.config import PlatypusConfig
from platypus.weights import chebyshev, random_weights
from platypus.errors import PlatypusError


class MOEAs(AbstractGeneticAlgorithm):

    def __init__(self, problem,
                 epsilons,
                 population_size,
                 generator,
                 selector,
                 variator=None,
                 **kwargs):
        super().__init__(problem, population_size, generator, **kwargs)
        self.variator = variator
        self.selector = selector
        self.archive = EpsilonBoxArchive(epsilons)
        self.population = None

    def step(self):
        if self.nfe == 0:
            self.initialize()
            self.result = self.archive
        else:
            self.iterate()
            self.result = self.archive

    def initialize(self):
        super().initialize()

        self.archive += self.population

        if self.variator is None:
            self.variator = PlatypusConfig.default_variator(self.problem)

    def iterate(self):
        pass


class NSGAII(MOEAs):

    def __init__(self, problem,
                 epsilons,
                 population_size=100,
                 generator=RandomGenerator(),
                 selector=TournamentSelector(2),
                 variator=None,
                 **kwargs):
        super().__init__(problem, epsilons, population_size, generator, selector, variator, **kwargs)

    def iterate(self):
        offspring = []

        while len(offspring) < self.population_size:
            parents = self.selector.select(self.variator.arity, self.population)
            offspring.extend(self.variator.evolve(parents))

        self.evaluate_all(offspring)

        offspring.extend(self.population)
        nondominated_sort(offspring)
        self.population = nondominated_truncate(offspring, self.population_size)

        self.archive.extend(self.population)


class IBEA(MOEAs):

    def __init__(self, problem,
                 epsilons,
                 population_size=100,
                 generator=RandomGenerator(),
                 fitness_evaluator=HypervolumeFitnessEvaluator(),
                 fitness_comparator=AttributeDominance(fitness_key, False),
                 variator=None,
                 **kwargs):
        super().__init__(problem, epsilons, population_size, generator,
                         TournamentSelector(2, fitness_comparator),
                         variator, **kwargs)
        self.fitness_evaluator = fitness_evaluator
        self.fitness_comparator = fitness_comparator

    def initialize(self):
        super().initialize()
        self.fitness_evaluator.evaluate(self.population)

    def iterate(self):
        offspring = []

        while len(offspring) < self.population_size:
            parents = self.selector.select(self.variator.arity, self.population)
            offspring.extend(self.variator.evolve(parents))

        self.evaluate_all(offspring)

        self.population.extend(offspring)
        self.fitness_evaluator.evaluate(self.population)

        while len(self.population) > self.population_size:
            self.fitness_evaluator.remove(self.population, self._find_worst())

        self.archive.extend(self.population)

    def _find_worst(self):
        index = 0

        for i in range(1, len(self.population)):
            if self.fitness_comparator.compare(self.population[index], self.population[i]) < 0:
                index = i

        return index


class MOEAD(MOEAs):

    def __init__(self, problem,
                 epsilons,
                 neighborhood_size=10,
                 generator=RandomGenerator(),
                 variator=None,
                 delta=0.8,
                 eta=1,
                 update_utility=None,
                 weight_generator=random_weights,
                 scalarizing_function=chebyshev,
                 **kwargs):
        super().__init__(problem, epsilons=epsilons, population_size=0,
                         generator=generator, selector=None,
                         variator=variator,
                         **remove_keys(kwargs, "population_size"))  # population_size is set after generating weights
        self.neighborhood_size = neighborhood_size
        self.variator = variator
        self.delta = delta
        self.eta = eta
        self.update_utility = update_utility
        self.weight_generator = weight_generator
        self.scalarizing_function = scalarizing_function
        self.generation = 0
        self.weight_generator_kwargs = only_keys_for(kwargs, weight_generator)

        # MOEA/D currently only works on minimization problems
        if any([d != Direction.MINIMIZE for d in problem.directions]):
            raise PlatypusError("MOEAD currently only works with minimization problems")

    def _update_ideal(self, solution):
        for i in range(self.problem.nobjs):
            self.ideal_point[i] = min(self.ideal_point[i], solution.objectives[i])

    def _calculate_fitness(self, solution, weights):
        return self.scalarizing_function(solution, self.ideal_point, weights)

    def _update_solution(self, solution, mating_indices):
        c = 0
        random.shuffle(mating_indices)

        for i in mating_indices:
            candidate = self.population[i]
            weights = self.weights[i]
            replace = False

            if solution.constraint_violation > 0.0 and candidate.constraint_violation > 0.0:
                if solution.constraint_violation < candidate.constraint_violation:
                    replace = True
            elif candidate.constraint_violation > 0.0:
                replace = True
            elif solution.constraint_violation > 0.0:
                pass
            elif self._calculate_fitness(solution, weights) < self._calculate_fitness(candidate, weights):
                replace = True

            if replace:
                self.population[i] = solution
                c = c + 1

            if c >= self.eta:
                break

    def _sort_weights(self, base, weights):
        def compare(weight1, weight2):
            dist1 = math.sqrt(sum([math.pow(base[i]-weight1[1][i], 2.0) for i in range(len(base))]))
            dist2 = math.sqrt(sum([math.pow(base[i]-weight2[1][i], 2.0) for i in range(len(base))]))

            if dist1 < dist2:
                return -1
            elif dist1 > dist2:
                return 1
            else:
                return 0

        sorted_weights = sorted(enumerate(weights), key=functools.cmp_to_key(compare))
        return [i[0] for i in sorted_weights]

    def initialize(self):
        self.population = []

        # initialize weights
        self.weights = self.weight_generator(self.problem.nobjs, **self.weight_generator_kwargs)
        self.population_size = len(self.weights)

        # initialize the neighborhoods based on weights
        self.neighborhoods = []

        for i in range(self.population_size):
            sorted_weights = self._sort_weights(self.weights[i], self.weights)
            self.neighborhoods.append(sorted_weights[:self.neighborhood_size])

        # initialize the ideal point
        self.ideal_point = [POSITIVE_INFINITY]*self.problem.nobjs

        # initialize the utilities and fitnesses
        self.utilities = [1.0]*self.population_size
        self.fitnesses = [0.0]*self.population_size

        # generate and evaluate the initial population
        self.population = [self.generator.generate(self.problem) for _ in range(self.population_size)]
        self.evaluate_all(self.population)

        # update the ideal point
        for i in range(self.population_size):
            self._update_ideal(self.population[i])

        # compute fitness
        for i in range(self.population_size):
            self.fitnesses[i] = self._calculate_fitness(self.population[i], self.weights[i])

        self.archive += self.population

        # set the default variator if one is not provided
        if self.variator is None:
            self.variator = PlatypusConfig.default_variator(self.problem)

    def _get_subproblems(self):
        indices = []

        if self.update_utility is None:
            indices.extend(list(range(self.population_size)))
        else:
            indices = []

            if self.weight_generator == random_weights:
                indices.extend(list(range(self.problem.nobjs)))

            while len(indices) < self.population_size:
                index = random.randrange(self.population_size)

                for _ in range(9):
                    temp_index = random.randrange(self.population_size)

                    if self.utilities[temp_index] > self.utilities[index]:
                        index = temp_index

                indices.append(index)

        random.shuffle(indices)
        return indices

    def _get_mating_indices(self, index):
        if random.uniform(0.0, 1.0) <= self.delta:
            return self.neighborhoods[index]
        else:
            return list(range(self.population_size))

    def _update_utility(self):
        for i in range(self.population_size):
            old_fitness = self.fitnesses[i]
            new_fitness = self._calculate_fitness(self.population[i], self.weights[i])
            relative_decrease = (old_fitness - new_fitness) / old_fitness

            if old_fitness - new_fitness > 0.001:
                self.utilities[i] = 1.0
            else:
                self.utilities[i] = min(1.0, 0.95 * (1.0 + 0.05*relative_decrease/0.001) * self.utilities[i])

            self.fitnesses[i] = new_fitness

    def iterate(self):
        for index in self._get_subproblems():
            mating_indices = self._get_mating_indices(index)
            parents = [self.population[index]] + [self.population[i] for i in mating_indices[:(self.variator.arity-1)]]
            offspring = self.variator.evolve(parents)

            self.evaluate_all(offspring)

            for child in offspring:
                self._update_ideal(child)
                self._update_solution(child, mating_indices)

        self.generation += 1

        if self.update_utility is not None and self.update_utility >= 0 and self.generation % self.update_utility == 0:
            self._update_utility()

        self.archive.extend(self.population)