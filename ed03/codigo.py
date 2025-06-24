#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Código: ga_tsp.py

Implementação de Algoritmo Genético para o Problema do Caixeiro Viajante (TSP)
usando a biblioteca DEAP. Suporta instâncias nos formatos TSPLIB (.tsp) e CSV (.csv).
Permite variação de operadores de crossover, taxa de mutação,
inicialização da população (aleatória vs. heurística) e critério de parada
(número fixo de gerações ou convergência).

Uso:
$ python ga_tsp.py --instance file/tsp_1.csv --crossover two_point --mut_rate 0.1 \
    --init_heur --converge --max_gen 100 --pop_size 100 --stall_gen 20
"""

import os
import random
import math
import time
import argparse
import csv
from deap import base, creator, tools


def read_tsp(file_path):
    """
    Lê um arquivo TSPLIB (.tsp) ou CSV (.csv) e retorna uma lista de coordenadas das cidades.
    CSV pode ter colunas "id,x,y" ou "x,y".
    """
    ext = os.path.splitext(file_path)[1].lower()
    cities = []
    # TSPLIB format
    if ext == '.tsp':
        coords = {}
        with open(file_path) as f:
            for line in f:
                if line.strip() == "NODE_COORD_SECTION":
                    break
            for line in f:
                if line.strip() in ("EOF", ""):
                    break
                parts = line.strip().split()
                if len(parts) >= 3:
                    idx = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    coords[idx] = (x, y)
        # Ordena pelas chaves (índices)
        cities = [coords[i] for i in sorted(coords.keys())]

    # CSV format
    elif ext == '.csv':
        with open(file_path, newline='') as f:
            reader = csv.reader(f)
            first_row = next(reader, None)
            if first_row is None:
                raise ValueError(f"Arquivo CSV vazio: {file_path}")

            # Detecta se a primeira linha é cabeçalho analisando se os dois
            # primeiros campos são números. Caso contrário, assume-se cabeçalho.
            try:
                float(first_row[0])
                if len(first_row) > 1:
                    float(first_row[1])
                is_header = False
            except ValueError:
                is_header = True

            rows = reader if is_header else [first_row] + list(reader)
            for row in rows:
                if not row:
                    continue
                # Se houver 3 colunas: id, x, y
                if len(row) >= 3:
                    x = float(row[1])
                    y = float(row[2])
                # Se houver 2 colunas: x, y
                else:
                    x = float(row[0])
                    y = float(row[1])
                cities.append((x, y))
    else:
        raise ValueError(f"Formato de arquivo não suportado: {ext}")

    if not cities:
        raise ValueError(f"Nenhuma cidade lida de: {file_path}")
    return cities


def euclidean_distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def total_distance(individual, cities):
    """Calcula a distância total do tour representado por `individual`."""
    dist = 0.0
    for i in range(len(individual)):
        start = cities[individual[i]]
        end = cities[individual[(i + 1) % len(individual)]]
        dist += euclidean_distance(start, end)
    return dist,


def cx_one_point_tsp(ind1, ind2):
    """Crossover de um ponto adaptado para permutações (TSP)."""
    size = len(ind1)
    cxpoint = random.randrange(1, size)
    p1, p2 = ind1[:], ind2[:]
    child1 = p1[:cxpoint] + [item for item in p2 if item not in p1[:cxpoint]]
    child2 = p2[:cxpoint] + [item for item in p1 if item not in p2[:cxpoint]]
    ind1[:] = child1
    ind2[:] = child2
    return ind1, ind2


def setup_ga(crossover_type, mut_rate, init_heuristic, cities):
    """Configura toolbox do DEAP para operadores e inicialização."""
    # Criação de classes de indivíduo e fitness
    if not hasattr(creator, 'FitnessMin'):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, 'Individual'):
        creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()

    # Geração de indivíduo
    def gen_individual():
        ind = list(range(len(cities)))
        random.shuffle(ind)
        return ind

    def heuristic_individual():
        unvisited = set(range(len(cities)))
        current = random.choice(list(unvisited))
        tour = [current]
        unvisited.remove(current)
        while unvisited:
            next_city = min(
                unvisited,
                key=lambda city: euclidean_distance(cities[current], cities[city])
            )
            tour.append(next_city)
            unvisited.remove(next_city)
            current = next_city
        return tour

    # Registro de criadores
    if init_heuristic:
        toolbox.register("individual", tools.initIterate, creator.Individual, heuristic_individual)
    else:
        toolbox.register("individual", tools.initIterate, creator.Individual, gen_individual)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", total_distance, cities=cities)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Operadores de crossover
    if crossover_type == "one_point":
        toolbox.register("mate", cx_one_point_tsp)
    elif crossover_type == "two_point":
        toolbox.register("mate", tools.cxPartialyMatched)
    elif crossover_type == "uniform":
        toolbox.register("mate", tools.cxUniformPartialyMatched, indpb=0.5)

    # Operador de mutação
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=mut_rate)

    return toolbox


def run_ga(toolbox, population, cxpb, mutpb, max_gen, stall_gen=None):
    """Executa o loop principal do AG com parada opcional por convergência."""
    # Avaliação inicial
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    best = min(population, key=lambda ind: ind.fitness.values[0])
    best_dist = best.fitness.values[0]
    stall = 0
    gen = 0

    while gen < max_gen:
        gen += 1
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for i in range(1, len(offspring), 2):
            if random.random() < cxpb:
                toolbox.mate(offspring[i-1], offspring[i])
                del offspring[i-1].fitness.values
                del offspring[i].fitness.values

        # Mutação
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Reavaliação
        invalids = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalids:
            ind.fitness.values = toolbox.evaluate(ind)

        population[:] = offspring

        # Atualiza melhor
        current_best = min(population, key=lambda ind: ind.fitness.values[0])
        curr_dist = current_best.fitness.values[0]
        if curr_dist < best_dist:
            best_dist = curr_dist
            best = current_best
            stall = 0
        else:
            stall += 1

        if stall_gen is not None and stall >= stall_gen:
            break

    return best_dist, gen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GA para TSP usando DEAP")
    parser.add_argument("--instance", type=str, required=True, help="Caminho para o arquivo .tsp ou .csv")
    parser.add_argument("--crossover", choices=["one_point","two_point","uniform"], default="two_point")
    parser.add_argument("--mut_rate", type=float, default=0.1, help="Probabilidade de mutação (indpb)")
    parser.add_argument("--init_heur", action="store_true", help="Usar inicialização heurística")
    parser.add_argument("--max_gen", type=int, default=100, help="Número máximo de gerações")
    parser.add_argument("--pop_size", type=int, default=100, help="Tamanho da população")
    parser.add_argument("--converge", action="store_true", help="Usar critério de convergência")
    parser.add_argument("--stall_gen", type=int, default=20, help="Gerações sem melhoria para convergência")
    args = parser.parse_args()

    # Leitura de cidades
    cities = read_tsp(args.instance)

    # Configuração do GA
    toolbox = setup_ga(args.crossover, args.mut_rate, args.init_heur, cities)

    # Criação da população
    population = toolbox.population(n=args.pop_size)
    start = time.time()
    best_dist, gens = run_ga(
        toolbox, population,
        cxpb=0.8,
        mutpb=args.mut_rate,
        max_gen=args.max_gen,
        stall_gen=(args.stall_gen if args.converge else None)
    )
    elapsed = time.time() - start

    print(f"Crossover: {args.crossover}")
    print(f"Mutação: {args.mut_rate}")
    print(f"Instância: {args.instance}")
    print(f"Melhor distância: {best_dist:.2f} após {gens} gerações")
    print(f"Tempo de execução: {elapsed:.4f}s \n")
