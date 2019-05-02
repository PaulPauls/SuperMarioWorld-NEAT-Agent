import pickle

import numpy as np

import retro
import neat
import cv2

env = retro.make('SuperMarioWorld-Snes', 'YoshiIsland1')
env_reshape_tuple = ()
time_to_xpos_dict = {}


def reward_from_data(data):
    reward = 0
    reward += int(data['x_pos_player']/10)
    reward += data['score']
    reward += 10 * data['coins']
    reward += 5000 * data['midway_point_flag']
    return reward

def end_genome_train_condition(data):
    t = data['timer_hundreds']*100 + data['timer_tens']*10 + data['timer_ones']
    time_to_xpos_dict[t] = data['x_pos_player']
    if t+5 in time_to_xpos_dict.keys():
        if time_to_xpos_dict[t+5] == time_to_xpos_dict[t]:
            time_to_xpos_dict.clear()
            return True

    return False


def eval_genomes(genomes, genome_config):
    for genome_id, genome in genomes:
        nn = neat.nn.recurrent.RecurrentNetwork.create(genome, genome_config)
        genome.fitness = 0
        done = False

        # cv2.namedWindow("machine_view", cv2.WINDOW_NORMAL)
        input_prepoc = env.reset()

        while not done:
            env.render()

            input_prepoc = cv2.resize(input_prepoc, env_reshape_tuple)
            input_prepoc = cv2.cvtColor(input_prepoc, cv2.COLOR_BGR2GRAY)
            nn_input = np.ndarray.flatten(input_prepoc)

            # cv2.imshow("machine_view", input_prepoc)
            # cv2.waitKey(1)

            nn_output = nn.activate(nn_input)

            input_prepoc, reward, done, data = env.step(nn_output)

            # print(data)

            genome.fitness += reward

            if done or end_genome_train_condition(data) or data['lives'] == 3:
                done = True
                genome.fitness += reward_from_data(data)
                print("Genome_ID: {} \t\tFitness: {}".format(genome_id, genome.fitness))


if __name__ == "__main__":
    config = neat.Config(neat.DefaultGenome,
                         neat.DefaultReproduction,
                         neat.DefaultSpeciesSet,
                         neat.DefaultStagnation,
                         'neat-config')

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.StatisticsReporter())
    pop.add_reporter(neat.Checkpointer(1))

    in_x, in_y, _ = env.observation_space.shape
    env_reshape_tuple = (int(in_x/8), int(in_y/8))

    best_genome = pop.run(eval_genomes)

    with open('best_genome.pkl', 'wb') as output:
        pickle.dump(best_genome, output, 1)
