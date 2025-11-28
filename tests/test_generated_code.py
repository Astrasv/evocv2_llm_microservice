import random
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Configuration constants
DIMENSIONS = 10
POP_SIZE = 200
NGEN = 100
CXPB = 0.8
MUTPB = 0.05
PENALTY = 0.01
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# Define fitness and individual classes
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def evaluate(individual):
    num_selected = sum(individual)
    if num_selected == 0:
        return (0,) # Avoid division by zero issues or invalid states
    # Simple surrogate accuracy: increases with number of selected features
    accuracy = 0.5 + 0.5 * (num_selected / DIMENSIONS)
    fitness = accuracy - PENALTY * num_selected
    return (fitness,)

def mate(ind1, ind2):
    tools.cxUniform(ind1, ind2, indpb=0.5)
    return ind1, ind2

def mutate(individual):
    tools.mutFlipBit(individual, indpb=MUTPB)
    return (individual,)

def select(population, k):
    return tools.selRoulette(population, k)

def create_individual():
    return [random.randint(0, 1) for _ in range(DIMENSIONS)]

toolbox = base.Toolbox()

# --- FIX 1: Correct Registration syntax (pass args separately) ---
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# --- FIX 2: Register the genetic operators ---
toolbox.register("evaluate", evaluate)
toolbox.register("mate", mate)
toolbox.register("mutate", mutate)
toolbox.register("select", select)

def main():
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("min", np.min)
    
    # verbose=True prints the logs to console
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB,
                                       ngen=NGEN, stats=stats, halloffame=hof,
                                       verbose=True)
    return pop, logbook, hof

if __name__ == "__main__":
    pop, logbook, hof = main()

    best_ind = hof[0]
    best_fitness = best_ind.fitness.values[0]
    num_features = sum(best_ind)
    
    print(f"\nBest Individual: {best_ind}")
    print(f"Fitness: {best_fitness:.4f}")
    print(f"Number of selected features: {num_features}")

    # Plot evolution of fitness
    gen = logbook.select("gen")
    avg = logbook.select("avg")
    max_f = logbook.select("max")
    
    plt.figure(figsize=(8,5))
    plt.plot(gen, avg, label='Average Fitness')
    plt.plot(gen, max_f, label='Max Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Evolution of Fitness over Generations')
    plt.legend()
    plt.grid(True)
    plt.show()