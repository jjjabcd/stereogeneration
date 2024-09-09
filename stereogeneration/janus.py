import os, sys
import multiprocessing
import random
import yaml
from functools import partial
from collections import OrderedDict
from typing import Callable, List, Optional

import numpy as np

from .crossover import crossover_smiles
from .mutate import mutate_smiles
from .network import create_and_train_network, obtain_model_pred
from .utils import sanitize_smiles, get_fp_scores, assign_stereo, neutralize_radicals, scramble_stereo
from .fragment import form_fragments


class JANUS:
    """ JANUS class for genetic algorithm applied on SELFIES
    string representation.
    See example/example.py for descriptions of parameters
    """

    def __init__(
        self,
        work_dir: str,
        fitness_function: Callable,
        start_population: str,
        verbose_out: Optional[bool] = False,
        custom_filter: Optional[Callable] = None,
        alphabet: Optional[List[str]] = None,
        alphabet_weights: Optional[List[float]] = None,
        use_gpu: Optional[bool] = True,
        num_workers: Optional[int] = None,
        generations: Optional[int] = 200,
        generation_size: Optional[int] = 5000,
        num_exchanges: Optional[int] = 5,
        use_fragments: Optional[bool] = True,
        num_sample_frags: Optional[int] = 200,
        use_classifier: Optional[bool] = True,
        explr_num_random_samples: Optional[int] = 5,
        explr_num_mutations: Optional[int] = 5,
        crossover_num_random_samples: Optional[int] = 1,
        exploit_num_random_samples: Optional[int] = 400,
        exploit_num_mutations: Optional[int] = 400,
        top_mols: Optional[int] = 1,
        stereo: bool = True,
        starting_fitness = None,
        use_diverse_topk = True,
    ):

        # set all class variables
        self.work_dir = work_dir
        self.fitness_function = fitness_function
        self.start_population = start_population
        self.verbose_out = verbose_out
        self.custom_filter = custom_filter
        self.alphabet = alphabet
        self.alphabet_weights = alphabet_weights
        self.use_gpu = use_gpu
        self.num_workers = num_workers if num_workers is not None else multiprocessing.cpu_count()
        self.generations = generations
        self.generation_size = generation_size
        self.num_exchanges = num_exchanges
        self.use_fragments = use_fragments
        self.num_sample_frags = num_sample_frags
        self.use_classifier = use_classifier
        self.explr_num_random_samples = explr_num_random_samples
        self.explr_num_mutations = explr_num_mutations
        self.crossover_num_random_samples = crossover_num_random_samples
        self.exploit_num_random_samples = exploit_num_random_samples
        self.exploit_num_mutations = exploit_num_mutations
        self.top_mols = top_mols
        self.stereo = stereo
        self.use_diverse_topk = use_diverse_topk

        # create dump folder
        if not os.path.isdir(f"./{self.work_dir}"):
            os.mkdir(f"./{self.work_dir}")
        self.save_hyperparameters()

        # get initial population
        init_smiles, init_fitness = [], []
        with open(self.start_population, "r") as f:
            for line in f:
                line = sanitize_smiles(line.strip())
                if line is not None:
                    init_smiles.append(line)
        # init_smiles = list((init_smiles))  # no duplicates

        # check that parameters are valid
        assert (
            len(init_smiles) >= self.generation_size
        ), "Initial population smaller than generation size."
        assert (
            self.top_mols <= self.generation_size
        ), "Number of top molecules larger than generation size."
        if self.num_exchanges > 0.1*self.generation_size:
            print('Parameter num_exchanges is greater than 0.1 of generation size. \
                  You may exchange too many molecules after each generation.')
            

        # make fragments from initial smiles
        self.frag_alphabet = []
        if self.use_fragments:
            if self.stereo:
                expanded_smi = []
                for smi in init_smiles:
                    expanded_smi.extend(scramble_stereo(smi))
            else:
                expanded_smi = init_smiles
                
            with multiprocessing.Pool(self.num_workers) as pool:
                frags = pool.map(
                    partial(
                        form_fragments,
                        stereo=self.stereo
                    ), expanded_smi)

            frags = self.flatten_list(frags)
            frags = list(set(frags))    # make it unique
            print(f"    Unique and valid fragments generated: {len(frags)}")
            self.frag_alphabet.extend(frags)


        # get initial fitness
        if starting_fitness is None:
            with multiprocessing.Pool(self.num_workers) as pool:
                init_fitness = pool.map(self.fitness_function, init_smiles)
        else:
            # use starting fitness 
            init_fitness = starting_fitness
            assert len(init_fitness) == len(init_smiles), 'Initial fitness should be same as smiles.'

        # sort the initial population and save in class
        idx = np.argsort(init_fitness)[::-1]
        init_smiles = np.array(init_smiles)[idx]
        init_fitness = np.array(init_fitness)[idx]
        self.population = init_smiles[: self.generation_size]
        self.fitness = init_fitness[: self.generation_size]

        with open(os.path.join(self.work_dir, "init_mols.txt"), "w") as f:
            f.writelines([f"{x}\n" for x in self.population])

        # store in collector, deal with duplicates
        self.smiles_collector = {}
        uniq_pop, idx, counts = np.unique(
            self.population, return_index=True, return_counts=True
        )
        for smi, count, i in zip(uniq_pop, counts, idx):
            self.smiles_collector[smi] = [self.fitness[i], count]


    def mutate_smi_list(self, smi_list: List[str], space="local"):
        # parallelized mutation function
        if space == "local":
            num_random_samples = self.exploit_num_random_samples
            num_mutations = self.exploit_num_mutations
        elif space == "explore":
            num_random_samples = self.explr_num_random_samples
            num_mutations = self.explr_num_mutations
        else:
            raise ValueError('Invalid space, choose "local" or "explore".')
        
        smi_list = smi_list * num_random_samples
            
        with multiprocessing.Pool(self.num_workers) as pool:
            mut_smi_list = pool.map(
                partial(
                    mutate_smiles,
                    alphabet=self.frag_alphabet,
                    num_random_samples=1,
                    num_mutations=num_mutations,
                    num_sample_frags=self.num_sample_frags,
                    base_alphabet=self.alphabet,
                    alphabet_weights=self.alphabet_weights,
                    stereo=self.stereo
                ),
                smi_list,
            )
        mut_smi_list = self.flatten_list(mut_smi_list)

        return mut_smi_list

    def crossover_smi_list(self, smi_list: List[str]):
        # parallelized crossover function
        with multiprocessing.Pool(self.num_workers) as pool:
            cross_smi = pool.map(
                partial(
                    crossover_smiles,
                    crossover_num_random_samples=self.crossover_num_random_samples,
                    stereo=self.stereo
                ),
                smi_list,
            )
        cross_smi = self.flatten_list(cross_smi)
        return cross_smi

    def measure_smi_list(self, smi_list: List[str]):
        with multiprocessing.Pool(self.num_workers) as pool:
            fitness = pool.map(
                self.fitness_function,
                smi_list
            )
        return fitness

    def check_filters(self, smi_list: List[str]):
        if self.custom_filter is not None:
            smi_list = [smi for smi in smi_list if self.custom_filter(smi)]
        return smi_list

    def neutralize_radicals(self, smi_list: List[str]):
        new_smi_list = []
        for smi in smi_list:
            new_smi = neutralize_radicals(smi)
            if new_smi is not None:
                new_smi_list.append(new_smi)
        return new_smi_list

    def save_hyperparameters(self):
        hparams = {
            k: v if not callable(v) else v.__name__ for k, v in vars(self).items()
        }
        with open(os.path.join(self.work_dir, "hparams.yml"), "w") as f:
            yaml.dump(hparams, f)

    def run(self):
        """ Run optimization based on hyperparameters initialized
        """

        # create some file with headers for output
        for fname in [
            'exploration_results.csv',
            'exploitation_results.csv',
            'generation_all_best.csv'
        ]:
            with open(os.path.join(self.work_dir, fname), 'a+') as f:
                f.writelines([f"generation,smiles,fitness\n"])

        for gen_ in range(self.generations):

            print(f"On generation {gen_}/{self.generations}")

            keep_smiles, replace_smiles = self.get_good_bad_smiles(
                self.fitness, self.population, self.generation_size
            )
            replace_smiles = list(set(replace_smiles))

            ### EXPLORATION ###
            # Mutate and crossover (with keep_smiles) molecules that are meant to be replaced
            explr_smiles = []
            timeout_counter = 0
            while len(explr_smiles) < self.generation_size-len(keep_smiles):
                # Mutations:
                mut_smi_explr = self.mutate_smi_list(
                    replace_smiles[0 : len(replace_smiles) // 2], space="explore"
                )
                mut_smi_explr = self.neutralize_radicals(mut_smi_explr)
                mut_smi_explr = self.check_filters(mut_smi_explr)

                # Crossovers:
                smiles_join = []
                for item in replace_smiles[len(replace_smiles) // 2 :]:
                    smiles_join.append(item + "xxx" + random.choice(keep_smiles))
                cross_smi_explr = self.crossover_smi_list(smiles_join)
                cross_smi_explr = self.neutralize_radicals(cross_smi_explr)
                cross_smi_explr = self.check_filters(cross_smi_explr)

                # Combine and get unique smiles not yet found
                all_smiles = list(set(mut_smi_explr + cross_smi_explr))
                copy_collector = self.smiles_collector.copy()
                for x in all_smiles:
                    smi = assign_stereo(x, copy_collector) if self.stereo else x
                    if smi not in self.smiles_collector:
                        explr_smiles.append(smi)
                # explr_smiles = list(set(explr_smiles))

                # print(f'Explore: {timeout_counter}    Len of unique smiles: {len(explr_smiles)}')
                print(f'Explore: {timeout_counter}    Len of smiles: {len(explr_smiles)}')
                print(f'\t\t Number of uniques {len(list(set(explr_smiles)))}')

                timeout_counter += 1
                if timeout_counter % 50 == 0:
                    print(f'Exploration: {timeout_counter} iterations of filtering. \
                    Filter may be too strict, or you need more mutations/crossovers.')

            # Replace the molecules with ones in exploration mutation/crossover
            if not self.use_classifier or gen_ == 0:
                replaced_pop = random.sample(
                    explr_smiles, self.generation_size - len(keep_smiles)
                )
            else:
                # The sampling needs to be done by the neural network!
                print("    Training classifier neural net...")
                train_smiles, targets = [], []
                for item in self.smiles_collector:
                    train_smiles.append(item)
                    targets.append(self.smiles_collector[item][0])
                net = create_and_train_network(
                    train_smiles,
                    targets,
                    num_workers=self.num_workers,
                    use_gpu=self.use_gpu,
                )

                # Obtain predictions on unseen molecules:
                print("    Obtaining Predictions")
                new_predictions = obtain_model_pred(
                    explr_smiles,
                    net,
                    num_workers=self.num_workers,
                    use_gpu=self.use_gpu,
                )
                sorted_idx = np.argsort(np.squeeze(new_predictions))[::-1]
                replaced_pop = np.array(explr_smiles)[
                    sorted_idx[: self.generation_size - len(keep_smiles)]
                ].tolist()

            # Calculate actual fitness for the exploration population
            self.population = keep_smiles + replaced_pop

            # get fitnesses already known
            self.fitness = []
            for smi in keep_smiles:
                self.fitness.append(self.smiles_collector[smi][0])
                self.smiles_collector[smi][1] += 1
            
            # calculate new fitnesses
            new_fitness = self.measure_smi_list(replaced_pop)
            self.fitness.extend(new_fitness)
            for smi, f in zip(replaced_pop, new_fitness):
                if smi in self.smiles_collector:
                    self.smiles_collector[smi][1] += 1
                else:
                    self.smiles_collector[smi] = [f, 1]

            # Print exploration data
            idx_sort = np.argsort(self.fitness)[::-1]
            print(f"    (Explr) Top Fitness: {self.fitness[idx_sort[0]]}")
            print(f"    (Explr) Top Smile: {self.population[idx_sort[0]]}")

            fitness_sort = np.array(self.fitness)[idx_sort]
            population_sort = np.array(self.population)[idx_sort]

            if self.verbose_out:
                with open(os.path.join(self.work_dir, 'exploration_results.csv'), 'a+') as f:
                    f.writelines([f"{gen_},{smi_in_pop},{fit_in_pop}\n" for smi_in_pop, fit_in_pop in zip(population_sort, fitness_sort)])
            else:
                with open(os.path.join(self.work_dir, 'final_exploration_results.csv'), 'w') as f:
                    f.writelines(
                        ['generation,smiles,fitness\n'] +
                        [f"{gen_},{smi_in_pop},{fit_in_pop}\n" for smi_in_pop, fit_in_pop in zip(population_sort, fitness_sort)]
                    )

            ### EXPLOITATION ###
            # Conduct local search on top-n molecules from population, mutate and do similarity search
            exploit_smiles = []
            timeout_counter = 0
            while len(exploit_smiles) < self.generation_size:
                # smiles_local_search = population_sort[0 : self.top_mols].tolist()
                smiles_local_search = self.get_diverse_topk(population_sort) if self.use_diverse_topk else population_sort[0:self.top_mols].tolist()
                mut_smi_loc = self.mutate_smi_list(smiles_local_search, "local")
                mut_smi_loc = self.neutralize_radicals(mut_smi_loc)
                mut_smi_loc = self.check_filters(mut_smi_loc)

                # filter out molecules already found
                copy_collector = self.smiles_collector.copy()
                for x in mut_smi_loc:
                    smi = assign_stereo(x, copy_collector) if self.stereo else x
                    if smi not in self.smiles_collector:
                        exploit_smiles.append(smi)
                # exploit_smiles = list(set(exploit_smiles))

                print(f'Exploit: {timeout_counter}    Len of unique smiles: {len(exploit_smiles)}')
                print(f'\t\t Number of uniques {len(list(set(exploit_smiles)))}')
                timeout_counter += 1
                if timeout_counter % 50 == 0:
                    print(f'Exploitation: {timeout_counter} iterations of filtering. \
                    Filter may be too strict, or you need more mutations/crossovers.')

            # sort by similarity, only keep ones similar to best
            fp_scores = get_fp_scores(exploit_smiles, population_sort[0])
            fp_sort_idx = np.argsort(fp_scores)[::-1][: self.generation_size]
            # highest fp_score idxs
            self.population_loc = np.array(exploit_smiles)[
                fp_sort_idx
            ]  # list of smiles with highest fp scores (most to least similar)

            # STEP 4: CALCULATE THE FITNESS FOR THE LOCAL SEARCH:
            # Exploitation data generated from similarity search is measured with fitness function

            # calculate new fitnesses
            self.fitness_loc = self.measure_smi_list(self.population_loc)
            for smi, f in zip(self.population_loc, self.fitness_loc):
                if smi in self.smiles_collector:
                    self.smiles_collector[smi][1] += 1
                else:
                    self.smiles_collector[smi] = [f, 1]


            # List of original local fitness scores
            idx_sort = np.argsort(self.fitness_loc)[
                ::-1
            ]  # index of highest to lowest fitness scores
            print(f"    (Local) Top Fitness: {self.fitness_loc[idx_sort[0]]}")
            print(f"    (Local) Top Smile: {self.population_loc[idx_sort[0]]}")

            fitness_sort = np.array(self.fitness_loc)[idx_sort]
            population_sort = np.array(self.population_loc)[idx_sort]

            if self.verbose_out:
                with open(os.path.join(self.work_dir, 'exploitation_results.csv'), 'a+') as f:
                    f.writelines([f"{gen_},{smi_in_pop},{fit_in_pop}\n" for smi_in_pop, fit_in_pop in zip(population_sort, fitness_sort)])
            else:
                with open(os.path.join(self.work_dir, 'final_exploitation_results.csv'), 'w') as f:
                    f.writelines(
                        ['generation,smiles,fitness\n'] +
                        [f"{gen_},{smi_in_pop},{fit_in_pop}\n" for smi_in_pop, fit_in_pop in zip(population_sort, fitness_sort)]
                    )

            # STEP 5: EXCHANGE THE POPULATIONS:
            # Introduce changes to 'fitness' & 'population'
            best_smi_local = population_sort[0 : self.num_exchanges]
            best_fitness_local = fitness_sort[0 : self.num_exchanges]

            # But will print the best fitness values in file
            idx_sort = np.argsort(self.fitness)[
                ::-1
            ]  # sorted indices for the entire population
            worst_indices = idx_sort[
                -self.num_exchanges :
            ]  # replace worst ones with the best ones
            for i, idx in enumerate(worst_indices):
                self.population[idx] = best_smi_local[i]
                self.fitness[idx] = best_fitness_local[i]

            # Save best of generation!:
            fit_all_best = np.argmax(self.fitness)

            # write best molecule with best fitness
            with open(os.path.join(self.work_dir, "generation_all_best.csv"), "a+") as f:
                f.writelines(
                    f"{gen_},{self.population[fit_all_best]},{self.fitness[fit_all_best]}\n"
                )

    def get_diverse_topk(self, smi_list, sim_thresh=0.15):
        # function will return diverse set of top molecules
        # smi_list and fitness_list should be sorted by fitness largest -> smallest
        top_smis = smi_list[:self.top_mols]

        smi_list = np.array(smi_list)
        diverse_topk = []
        while len(diverse_topk) < self.top_mols:
            diverse_topk.append(smi_list[0])                            # add the top fitness
            scores = np.array(get_fp_scores(smi_list, smi_list[0]))     # get similarity scores

            smi_list = smi_list[scores < sim_thresh]       # remove the highest sim, also removes the top1
            if len(smi_list) < self.top_mols:
                print('Molecules in the list of smiles are too similar to find diverse topk.')
                return top_smis
        
        # if top_smis == diverse_topk:
        #     print('Diverse topk is the same as topk smiles.')
        # print(top_smis)
        # print(diverse_topk)

        return diverse_topk
        

    @staticmethod
    def get_good_bad_smiles(fitness, population, generation_size):
        """
        Given fitness values of all SMILES in population, and the generation size, 
        this function smplits  the population into two lists: keep_smiles & replace_smiles. 
        
        Parameters
        ----------
        fitness : (list of floats)
            List of floats representing properties for molecules in population.
        population : (list of SMILES)
            List of all SMILES in each generation.
        generation_size : (int)
            Number of molecules in each generation.

        Returns
        -------
        keep_smiles : (list of SMILES)
            A list of SMILES that will be untouched for the next generation. .
        replace_smiles : (list of SMILES)
            A list of SMILES that will be mutated/crossed-oved for forming the subsequent generation.

        """

        fitness = np.array(fitness)
        idx_sort = fitness.argsort()[::-1]  # Best -> Worst
        keep_ratio = 0.2
        keep_idx = int(len(list(idx_sort)) * keep_ratio)
        # try:

        #     F_50_val = fitness[idx_sort[keep_idx]]
        #     F_25_val = np.array(fitness) - F_50_val
        #     F_25_val = np.array([x for x in F_25_val if x < 0]) + F_50_val
        #     F_25_sort = F_25_val.argsort()[::-1]
        #     F_25_val = F_25_val[F_25_sort[0]]

        #     prob_ = 1.0 / (3.0 ** ((F_50_val - fitness) / (F_50_val - F_25_val)) + 1)

        #     # prob_ = prob_ / sum(prob_)
        #     to_keep = np.random.choice(generation_size, keep_idx, p=prob_, replace=False)
        #     to_replace = [i for i in range(generation_size) if i not in to_keep][
        #         0 : generation_size - len(to_keep)
        #     ]

        #     keep_smiles = [population[i] for i in to_keep]
        #     replace_smiles = [population[i] for i in to_replace]

        #     best_smi = population[idx_sort[0]]
        #     if best_smi not in keep_smiles:
        #         keep_smiles.append(best_smi)
        #         if best_smi in replace_smiles:
        #             replace_smiles.remove(best_smi)

        #     if keep_smiles == [] or replace_smiles == []:
        #         raise Exception("Badly sampled population!")
        # except:
        keep_smiles = [population[i] for i in idx_sort[:keep_idx]]
        replace_smiles = [population[i] for i in idx_sort[keep_idx:]]

        return keep_smiles, replace_smiles


    @staticmethod
    def flatten_list(nested_list):
        return [item for sublist in nested_list for item in sublist]

