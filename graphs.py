import matplotlib.pyplot as plt
import numpy as np
import math

def reward_vs_steps(in_path_arr):
    rewards = []
    steps = []
    max_points_at_graph = 25
    for in_path in in_path_arr:
        f = open(in_path)
        lines = f.readlines()
        f.close()
        for line in lines[:-1]:
            fields = line.split(' ')
            rewards.append(float(fields[-1]))
            steps.append(float(fields[1])/1000)
        jump_val = math.ceil(len(rewards)/max_points_at_graph)
        plt.plot(steps[0::jump_val],rewards[0::jump_val])
    plt.xlabel('steps [K]')
    plt.ylabel('reward (average over last 3k steps)')
    plt.title('Average reward vs number of steps')
    plt.show()

in_path_arr = ['final_results/no_rnn/lr_0.000010_batch_size:_32.000000_trajectory_length:_10.000000_flickering_p_0.000000_is_rnn:_False',
               'final_results/rnn_10/lr_0.000010_batch_size:_32.000000_trajectory_length:_10.000000_flickering_p_0.000000_is_rnn:_True']
reward_vs_steps(in_path_arr)
a = 0