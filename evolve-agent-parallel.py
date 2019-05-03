import os
import pickle

import numpy as np

import retro
import neat
import cv2

# Config Values of training program (not of NEAT)
# Config values of NEAT algorithm can be found in 'neat-config'
env = retro.make('SuperMarioWorld-Snes', 'YoshiIsland1')
max_generations = 300
cv_resolution_divisor = 8
nr_parallel_env = 4

# Global variable to keep program simple
env_reshape_tuple = ()


def reward_from_data(data):
    """ Convert meta information in data to a reward number. Reward focus is put on the x distance travelled and
        triggering special events like reaching the midway point flag """
    reward = 0
    reward += int(data['x_pos_player'])
    reward += int(0.1 * data['score'])
    reward += data['coins']
    reward += 5000 * data['midway_point_flag']
    return reward


def end_genome_train_condition(data, time_to_xpos_dict):
    """ Return true if 5 seconds of ingame time passed without the player changing its x position """
    t = data['timer_hundreds']*100 + data['timer_tens']*10 + data['timer_ones']
    time_to_xpos_dict[t] = data['x_pos_player']
    if t+5 in time_to_xpos_dict.keys():
        if time_to_xpos_dict[t+5] == time_to_xpos_dict[t]:
            time_to_xpos_dict.clear()
            return True

    return False


def eval_genome(genome, config):
    """ Train a single on the environment by letting it convert the computer vision input to button outputs
        and the evaluate and return (not save because parallel env!) its fitness according to reward_form_data """
    nn = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
    time_to_xpos_dict = {}
    fitness = 0
    done = False

    input_prepoc = env.reset()

    while not done:
        env.render()

        input_prepoc = cv2.resize(input_prepoc, env_reshape_tuple)
        input_prepoc = cv2.cvtColor(input_prepoc, cv2.COLOR_BGR2GRAY)
        nn_input = np.ndarray.flatten(input_prepoc)

        nn_output = nn.activate(nn_input)

        input_prepoc, reward, done, data = env.step(nn_output)

        if done or end_genome_train_condition(data, time_to_xpos_dict) or data['lives'] == 3:
            done = True
            fitness = reward_from_data(data)
            print("Genome_ID: {} \t\tFitness: {}".format("NA", fitness))

    return fitness


def run(config_file_path, checkpoint_file_path=None):
    """ Load Neat Default configuration, register reporters and train the population in parallel with eval_genome  """
    config = neat.Config(neat.DefaultGenome,
                         neat.DefaultReproduction,
                         neat.DefaultSpeciesSet,
                         neat.DefaultStagnation,
                         config_file_path)

    # Create the population, which is the top-level object for a NEAT run.
    pop = neat.Population(config)

    # Add reporters to show progress in the terminal.
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.StatisticsReporter())

    # Enable Checkpointing and possibly load existing checkpoint
    pop.add_reporter(neat.Checkpointer(5))
    if checkpoint_file_path is not None:
        print("Using pre-existing checkpoint and configuration! Possible different config file is dismissed!")
        pop = neat.Checkpointer.restore_checkpoint(checkpoint_file_path)

    # Calculate reshape resolution tuple according to specified parameters
    in_x, in_y, _ = env.observation_space.shape
    global env_reshape_tuple
    env_reshape_tuple = (int(in_x / cv_resolution_divisor), int(in_y / cv_resolution_divisor))

    # Run specified nr of parallel trainings for up to a specified max number of generations.
    pe = neat.ParallelEvaluator(nr_parallel_env, eval_genome)
    best_genome = pop.run(pe.evaluate, max_generations)

    # Display and save the winning genome.
    print('\nBest genome:\n{!s}'.format(best_genome))
    with open('best_genome.pkl', 'wb') as output:
        pickle.dump(best_genome, output, 1)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat-config')
    run(config_path)
