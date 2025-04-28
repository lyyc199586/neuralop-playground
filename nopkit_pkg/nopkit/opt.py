"""
opt.py

This module implements a Genetic Algorithm (GA) for optimizing sensor placement
on a grid. The GA is designed to work with constraints and evaluate fitness
based on a provided model and dataset.

Author: Yangyuanchen Liu
Date: 2025-04-27

Classes:
    GeneticAlgorithm: Represents the GA for sensor placement optimization.

Functions:
    plot_individual: Visualizes the sensor placement on a grid.
"""

import random
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from neuralop.models import FNO
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from einops import repeat, rearrange
from pathlib import Path

from deap import base, creator, tools, algorithms


def plot_individual(
    individual: List[Tuple[int, int]],
    grid_size_x: int = 32,
    grid_size_y: int = 32,
    save_path: Optional[str] = None,
) -> None:
    """
    Plots the sensor placement on a grid.

    Args:
        individual (List[Tuple[int, int]]): List of sensor positions as (y, x) tuples.
        grid_size_x (int): Width of the grid.
        grid_size_y (int): Height of the grid.
        save_path (Optional[str]): Path to save the plot. If None, the plot is not saved.

    Returns:
        None
    """

    layout = np.zeros((grid_size_y, grid_size_x))

    for y, x in individual:
        layout[y, x] = 1

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(
        layout, origin="lower", cmap="gray_r", extent=(0, grid_size_x, 0, grid_size_y)
    )

    ax.set_xticks([])
    ax.set_yticks([])

    if save_path:
        fig.savefig(save_path)

    plt.show()
    plt.close(fig)


def plot_best_layouts(
    history_dir: str, generations: list, grid_size_x: int = 32, grid_size_y: int = 32
):
    """
    Plots best sensor layouts from saved generation JSON files.

    Args:
        history_dir (str): Directory where generation JSON files are stored.
        generations (list of int): List of generations to plot.
        grid_size_x (int): Grid width.
        grid_size_y (int): Grid height.
    """
    history_dir = Path(history_dir)
    fig, axes = plt.subplots(1, len(generations), figsize=(3 * len(generations), 3))

    if len(generations) == 1:
        axes = [axes]

    for ax, gen in zip(axes, generations):
        file = history_dir / f"gen{gen}.json"
        with open(file, "r") as f:
            data = json.load(f)

        best_individual = list(zip(data[0]["y"], data[0]["x"]))  # first one is best

        layout = np.zeros((grid_size_y, grid_size_x))
        for y, x in best_individual:
            layout[y, x] = 1

        ax.imshow(
            layout,
            origin="lower",
            cmap="gray_r",
            extent=(0, grid_size_x, 0, grid_size_y),
        )
        ax.set_title(f"Gen {gen}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def plot_fitness_curve(
    history_dir: str, ax=None, plot_best=True, plot_avg=False, plot_worst=False
):
    """
    Plots fitness curves over generations.

    Args:
        history_dir (str): Directory where generation JSON files are stored.
        plot_best (bool): Whether to plot best fitness.
        plot_avg (bool): Whether to plot average fitness.
        plot_worst (bool): Whether to plot worst fitness.
    """
    history_dir = Path(history_dir)

    best_fitness = []
    avg_fitness = []
    worst_fitness = []
    generations = []

    json_files = sorted(
        history_dir.glob("gen*.json"), key=lambda p: int(p.stem.replace("gen", ""))
    )

    for file in json_files:
        with open(file, "r") as f:
            data = json.load(f)

        fitnesses = [ind["fitness"] for ind in data]
        generations.append(int(file.stem.replace("gen", "")))

        best_fitness.append(min(fitnesses))
        avg_fitness.append(sum(fitnesses) / len(fitnesses))
        worst_fitness.append(max(fitnesses))

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 2))

    if plot_best:
        ax.plot(generations, best_fitness, label="Best Fitness", marker="o")
    if plot_avg:
        ax.plot(generations, avg_fitness, label="Average Fitness", linestyle="--")
    if plot_worst:
        ax.plot(generations, worst_fitness, label="Worst Fitness", linestyle=":")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    # plt.title("Fitness Curve over Generations")
    ax.legend()
    # plt.tight_layout()
    plt.show()


class GeneticAlgorithm:
    """
    A class to represent a Genetic Algorithm for optimizing sensor placement with an FNO model.

    Attributes:
        test_db (torch.utils.data.Dataset): The dataset for evaluation.
        model (FNO): The Fourier Neural Operator model.
        data_processor (DefaultDataProcessor): Data processor for preprocessing and postprocessing.
        device (str): Device to run the computations ('cpu' or 'cuda').
        num_sensors_range (Tuple[int, int]): Range for the number of sensors.
        grid_size_x (int): Width of the grid.
        grid_size_y (int): Height of the grid.
        lambda_factor (float): Regularization factor for the fitness function.
        pop_size (int): Population size.
        gen_size (int): Number of generations.
        cxpb (float): Crossover probability.
        mutpb (float): Mutation probability.
        toolbox (base.Toolbox): DEAP toolbox for GA operations.
    """

    def __init__(
        self,
        test_db: torch.utils.data.Dataset,
        model: FNO,
        data_processor: DefaultDataProcessor,
        device: str = "cpu",
        num_sensors_range: Tuple[int, int] = (5, 17),
        # initial_individual: Optional[List[Tuple[int, int]]] = None,
        grid_size_x: int = 32,
        grid_size_y: int = 32,
        lambda_factor: float = 0.05,
        pop_size: int = 50,
        gen_size: int = 100,
        cxpb: float = 0.8,
        mutpb: float = 0.2,
    ) -> None:
        """
        Constructs all the necessary attributes for the GeneticAlgorithm object.

        Args:
            test_db (torch.utils.data.Dataset): The dataset for evaluation.
            model (FNO): The Fourier Neural Operator model.
            data_processor (DefaultDataProcessor): Data processor for preprocessing and postprocessing.
            device (str): Device to run the computations ('cpu' or 'cuda').
            num_sensors_range (Tuple[int, int]): Range for the number of sensors.
            grid_size_x (int): Width of the grid.
            grid_size_y (int): Height of the grid.
            lambda_factor (float): Regularization factor for the fitness function.
            pop_size (int): Population size.
            gen_size (int): Number of generations.
            cxpb (float): Crossover probability.
            mutpb (float): Mutation probability.

        Returns:
            None
        """
        # data and model
        self.test_db = test_db
        self.model = model
        self.data_processor = data_processor
        self.device = device
        self.data_processor.to(self.device)
        self.data_processor.eval()  # only use to eval

        # grid size and sensor constraints
        self.num_sensors_range = num_sensors_range
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y

        # GA parameters
        self.lambda_factor = lambda_factor
        self.pop_size = pop_size
        self.gen_size = gen_size
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.first_gen = True
        # self.initial_individual = initial_individual
        self.toolbox = base.Toolbox()
        self.setup_ga()

    def init_individual(self) -> List[Tuple[int, int]]:
        """
        Initializes an individual with constraints:
        1. Grids (the same with spatial mesh in x and y).
        2. Range for number of sensors (5-17).
        3. One sensor at the top center (31, 16).
        4. All other sensors on the right boundary (..., 31).
        5. No adjacent sensors.

        Returns:
            List[Tuple[int, int]]: A list of sensor positions as (y, x) tuples.
        """
        # if self.first_gen:
        #     num_sensors = random.randint(10, 17)
        # else:
        #     num_sensors = random.randint(
        #         self.num_sensors_range[0], self.num_sensors_range[1]
        #     )
        
        # if self.first_gen and self.initial_individual is not None:
        #     return self.initial_individual.copy()

        num_sensors = random.randint(
            self.num_sensors_range[0], self.num_sensors_range[1]
        )

        sensors = set()
        sensors.add((31, 16))  # add the center top sensor

        candidates = [y for y in range(self.grid_size_y) if y != 31]
        random.shuffle(candidates)

        for y in candidates:
            if len(sensors) >= num_sensors:
                break
            if (y - 1, 31) not in sensors and (y + 1, 31) not in sensors:
                sensors.add((y, 31))

        if len(sensors) < num_sensors:
            return self.init_individual()

        return list(sensors)

    def repair_individual(
        self, individual: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Repairs an individual after crossover or mutation to ensure constraints are met.

        Args:
            individual (List[Tuple[int, int]]): The individual to repair.

        Returns:
            List[Tuple[int, int]]: The repaired individual.
        """
        # ensure fixed sensor:
        has_center = (31, 16) in individual
        individual = [(y, x) for (y, x) in individual if x == 31]

        if not has_center:
            individual.append((31, 16))

        # remove adjacent sensors
        y_positions = sorted([y for (y, x) in individual])
        valid = []
        for y in y_positions:
            if not valid or abs(y - valid[-1][0]) > 1:
                valid.append((y, 31))

        if (31, 16) not in valid:
            valid.append((31, 16))

        # ensure number of sensors within range
        if len(individual) < self.num_sensors_range[0]:
            return self.init_individual()
        elif len(individual) > self.num_sensors_range[1]:
            # Keep (31,16) + randomly select others
            others = [pos for pos in valid if pos != (31, 16)]
            selected = random.sample(others, self.num_sensors_range[1] - 1)
            valid = [(31, 16)] + selected

        return valid

    def layout_to_mask(self, individual: List[Tuple[int, int]]) -> torch.Tensor:
        """
        Converts an individual layout to a binary mask.

        Args:
            individual (List[Tuple[int, int]]): The individual layout.

        Returns:
            torch.Tensor: A binary mask of the layout.
        """
        mask = torch.zeros((self.grid_size_y, self.grid_size_x), dtype=torch.float32)
        for y, x in individual:
            mask[y, x] = 1.0
        return mask.to(self.device)

    def loss(self, individual: List[Tuple[int, int]]) -> float:
        """
        Evaluates the loss of an individual.

        Args:
            individual (List[Tuple[int, int]]): The individual to evaluate.

        Returns:
            float: The loss value.
        """
        mask = self.layout_to_mask(individual)

        total_loss = 0.0
        num_elements = 0

        for data in self.test_db:
            x = data["x"].to(self.device)  # (c, h, w, t)
            y = data["y"].to(self.device)  # (c, h, w, t)

            # assemble new input with mask
            mask_expanded = repeat(mask, "x y -> 1 x y t", t=x.shape[-1])
            x_new = x.clone()
            x_new[0, :, :, :] = y[0, :, :, :] * mask_expanded
            x_new[1, :, :, :] = mask_expanded

            data_new = {"x": x_new, "y": y}
            data_new = self.data_processor.preprocess(data_new, batched=True)

            with torch.no_grad():
                pred = self.model(data_new["x"])

            pred, _ = self.data_processor.postprocess(pred, data_new)

            # print(f"y shape: {y.shape}, pred shape: {pred.shape}, mask shape: {mask_expanded.shape}")
            loss = torch.nn.functional.mse_loss(pred[0, 2, :, :, :], y[2, :, :, :])
            total_loss += loss.item()

            # accumulate number of elements
            num_elements += y.numel()

        return total_loss / num_elements

    def fitness(self, individual: List[Tuple[int, int]]) -> Tuple[float]:
        """
        Fitness is the average loss plus regularization on the number of sensors.

        Args:
            individual (List[Tuple[int, int]]): The individual to evaluate.

        Returns:
            Tuple[float]: The fitness value as a tuple.
        """
        num_sensors = len(individual)
        if num_sensors == 0:
            return 1e6

        pred_loss = self.loss(individual)
        fitness = pred_loss + self.lambda_factor * num_sensors
        return (fitness,)

    def setup_ga(self) -> None:
        """
        Sets up DEAP GA components, including individual creation, population
        initialization, selection, crossover, mutation, and evaluation.

        Returns:
            None
        """
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox.register(
            "individual", tools.initIterate, creator.Individual, self.init_individual
        )
        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.individual,
            n=self.pop_size,
        )

        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)

        self.toolbox.register("evaluate", self.fitness)

    def save_generation(self, pop, generation, hof):
        """
        Saves the entire population of a generation to a JSON file.
        The best individual is saved first.
        """
        filename = f"./history/gen{generation}.json"

        # Sort population: best individual first
        best = hof[0]
        individuals = [best] + [ind for ind in pop if ind != best]

        data = []
        for ind in individuals:
            record = {
                "x": [x for (y, x) in ind],
                "y": [y for (y, x) in ind],
                "fitness": ind.fitness.values[0],
            }
            data.append(record)

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    def run_ga(self) -> None:
        """
        Runs the genetic algorithm.

        Returns:
            None
        """
        pop = self.toolbox.population()
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        stats.register("avg", np.mean)

        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals", "min", "avg"]

        # Initial evaluation
        print("Initial evaluation...")
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        self.first_gen = False

        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=0, nevals=len(pop), **record)
        print(logbook.stream)
        self.save_generation(pop, 0, hof)

        # GA loop
        for gen in range(1, self.gen_size + 1):
            print(f"\nGeneration {gen} start...")

            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))

            # Elitism: keep best from previous generation
            best_prev = tools.selBest(pop, 1)[0]
            offspring[0] = self.toolbox.clone(best_prev)

            # crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cxpb:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # mutation
            for mutant in offspring:
                if random.random() < self.mutpb:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Repair all individuals
            for i in range(len(offspring)):
                repaired = self.repair_individual(offspring[i])
                offspring[i][:] = repaired

            # Evaluate
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            print(f"  Evaluating {len(invalid_ind)} offspring...")
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            pop[:] = offspring
            hof.update(pop)
            record = stats.compile(pop)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            print(logbook.stream)

            self.save_generation(pop, gen, hof)

        print("\n=== GA Finished ===")
        print("Best Sensor Layout:", hof[0])
