import numpy as np

import retro
import neat
import cv2
import pickle


def eval_genomes(genomes, genome_config):

    for genome_id, genome in genomes:
        ob = env.reset()

        print(env.action_space.sample())

        inx, iny, inc = env.observation_space.shape

        print(env.observation_space.shape)

        inx = int(inx/8)
        iny = int(iny/8)

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, genome_config)

        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        # xpos = 0
        # xpos_max = 0

        done = False

        while not done:
            env.render()
            frame += 1

            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))

            imgarray = np.ndarray.flatten(ob)

            nn_output = net.activate(imgarray)

            ob, rew, done, info = env.step(nn_output)

            # print("ob: {}".format(ob))
            # print("rew: {}".format(rew))
            # print("done: {}".format(done))
            # print("info: {}".format(info))

            fitness_current += rew

            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1

            if done or counter == 250:
                done = True
                print(genome_id, fitness_current)

            genome.fitness = fitness_current


if __name__ == "__main__":
    env = retro.make('SuperMarioWorld-Snes', 'DonutPlains1')

    config = neat.Config(neat.DefaultGenome,
                         neat.DefaultReproduction,
                         neat.DefaultSpeciesSet,
                         neat.DefaultStagnation,
                         'neat-config')

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    winner = p.run(eval_genomes)

    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)
