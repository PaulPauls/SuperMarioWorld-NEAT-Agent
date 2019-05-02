import pickle

import numpy as np

import retro
import neat
import cv2


class Worker(object):
    """ TODO """

    def __init__(self, genome, genome_config):
        self.genome = genome
        self.genome_config = genome_config
        self.env = retro.make('SuperMarioWorld-Snes', 'YoshiIsland1')
        self.env_reshape_tuple = ()
        self.time_to_xpos_dict = {}

        in_x, in_y, _ = self.env.observation_space.shape
        self.env_reshape_tuple = (int(in_x / 8), int(in_y / 8))

    @staticmethod
    def reward_from_data(data):
        reward = 0
        reward += int(data['x_pos_player'])
        reward += int(data['score'])
        reward += int(data['coins'])
        reward += 5000 * data['midway_point_flag']
        return reward

    def end_genome_train_condition(self, data):
        t = data['timer_hundreds'] * 100 + data['timer_tens'] * 10 + data['timer_ones']
        self.time_to_xpos_dict[t] = data['x_pos_player']
        if t + 5 in self.time_to_xpos_dict.keys():
            if self.time_to_xpos_dict[t + 5] == self.time_to_xpos_dict[t]:
                self.time_to_xpos_dict.clear()
                return True

        return False

    def work(self):

        nn = neat.nn.recurrent.RecurrentNetwork.create(self.genome, self.genome_config)
        fitness = 0
        done = False

        input_prepoc = self.env.reset()

        while not done:
            self.env.render()

            input_prepoc = cv2.resize(input_prepoc, self.env_reshape_tuple)
            input_prepoc = cv2.cvtColor(input_prepoc, cv2.COLOR_BGR2GRAY)
            nn_input = np.ndarray.flatten(input_prepoc)

            nn_output = nn.activate(nn_input)

            input_prepoc, reward, done, data = self.env.step(nn_output)

            fitness += reward

            if done or self.end_genome_train_condition(data) or data['lives'] == 3:
                done = True
                fitness += self.reward_from_data(data)
                print("Genome_ID: {} \t\tFitness: {}".format("NONE", fitness))

        return fitness


def eval_genomes(genome, genome_config):
    worker = Worker(genome, genome_config)
    return worker.work()


if __name__ == "__main__":
    config = neat.Config(neat.DefaultGenome,
                         neat.DefaultReproduction,
                         neat.DefaultSpeciesSet,
                         neat.DefaultStagnation,
                         'neat-config')

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.StatisticsReporter())
    pop.add_reporter(neat.Checkpointer(5))

    # If already trained some, load the current training checkpoint
    # pop = neat.Checkpointer.restore_checkpoint('neat-checkpoint-x')

    # Set up parallel evaluation
    thread_count = 10
    parallel_evaluator = neat.ParallelEvaluator(thread_count, eval_genomes)

    best_genome = pop.run(parallel_evaluator.evaluate)

    with open('best_genome.pkl', 'wb') as output:
        pickle.dump(best_genome, output, 1)
