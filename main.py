import train
import os
import copy

base_dir = os.path.abspath('results_2_fruit')
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

# these are our default params.
kwargs = {
    'grid_dim': 3,
    'num_of_obj': 2,
    'mem_capacity': 100000,
    'batch' : 128,
    'lr' : 0.001,
    'double_dqn' : False,
    'gamma' : 0.99,
    'num_steps' : 300000,
    'target_update_freq': 500,
    'learn_start' : 1500,
    'plot_update_freq' : 100,
    'eval_freq' : 500,
    'eval_episodes' : 10,
    'eps_decay' : 1000,
    'eps_end' : 0.1,
    'hidden_layer' : 100,
    'l1_regularization': 0,
    'dropout' : 0,
    'is_visdom' : False,
    'output_dir' : base_dir,
    'write_mode' : 'w',
}

is_run_grid_search =  True
is_run_best_results = True

if is_run_best_results:
    cur_kwargs = copy.deepcopy(kwargs)
    lr = 0.001
    batch = 128
    target_update_freq = 1500
    dir_name = '/best_config'
    output_dir =  (base_dir + dir_name + '/' + 'batch_'
    + str(batch) + '_' + '_lr_' + str(lr) + '_target_update_freq_' + str(target_update_freq))
    cur_kwargs['lr'] = lr
    cur_kwargs['batch'] = batch
    cur_kwargs['target_update_freq'] = target_update_freq
    cur_kwargs['output_dir'] = output_dir

    if not os.path.exists(base_dir + dir_name):
        os.mkdir(base_dir + dir_name)
    avg_rewards = []
    iter_num = 10
    for iter in range(iter_num):
        avg_rewards.append(train.run_train_session(**cur_kwargs))
        cur_kwargs['write_mode'] = 'a'
    f = open(output_dir,'a')
    f.write('Total average reward: ' + str(sum(avg_rewards)/float(len(avg_rewards))))
    f.close()

if is_run_grid_search:
    # run over learning rate
    lr_range = [0.1,0.01,0.001,0.0001,0.00001]
    dir_name = '/lr'
    if not os.path.exists(base_dir+dir_name):
        os.mkdir(base_dir+dir_name)

    for lr in lr_range:
        print('**********************************************\n')
        print('lr test, lr = %f \n' % lr)
        print('**********************************************\n')
        cur_kwargs = copy.deepcopy(kwargs)
        cur_kwargs['lr'] = lr
        cur_kwargs['output_dir'] = base_dir + dir_name + '/' + 'lr_' + str(lr)
        train.run_train_session(**cur_kwargs)

    # run over batch size
    batch_size_range = [32,64,128,256,512]
    dir_name = '/batch_size'
    if not os.path.exists(base_dir+dir_name):
        os.mkdir(base_dir+dir_name)

    for batch_size in batch_size_range:
        print('**********************************************\n')
        print('batch size test, batch size = %f \n' % batch_size)
        print('**********************************************\n')
        cur_kwargs = copy.deepcopy(kwargs)
        cur_kwargs['batch'] = batch_size
        cur_kwargs['output_dir'] = base_dir + dir_name + '/' + 'batch_' + str(batch_size)
        train.run_train_session(**cur_kwargs)

    # run over target network update frequency
    target_update_freq_range = [250,500,750,1000,1500]
    dir_name = '/target_update_freq'
    if not os.path.exists(base_dir+dir_name):
        os.mkdir(base_dir+dir_name)

    for update_freq in target_update_freq_range:
        print('**********************************************\n')
        print('target network update test, update frequency = %f \n' % update_freq)
        print('**********************************************\n')
        cur_kwargs = copy.deepcopy(kwargs)
        cur_kwargs['target_update_freq'] = update_freq
        cur_kwargs['output_dir'] = base_dir + dir_name + '/' + 'target_update_freq_' + str(update_freq)
        train.run_train_session(**cur_kwargs)






# import torch
# import torch.nn as nn
# from torch import optim
# import torch.nn.functional as F
# import gym
# import gym.spaces
# import numpy as np
# from collections import namedtuple
# import random
# from matplotlib import pyplot as pl
# from IPython.display import clear_output
# from DQN import DQN_MLP, ReplayBuffer, init_weights
# from GridWorldSimon import gameEnv
# from torch.autograd import Variable
# import matplotlib.pyplot as plt
#
#
#
# Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
# is_chen = False
# # is_chen = True
# #
# if is_chen:
#     env = gym.make('Taxi-v2')
#     eval_env = gym.make('Taxi-v2')
# else:
#     grid_dim = 3
#     num_of_obj = 1
#     env = gameEnv(size=grid_dim,startDelay=num_of_obj)
#     eval_env = gameEnv(size=grid_dim,startDelay=num_of_obj)
#
#
# input_size = env.observation_space.n
# output_size = env.action_space.n
# mem_capacity = 1000000
# batch = 128
# lr = 0.001
# double_dqn = False
# gamma = 0.99
# num_steps = 1000000
# target_update_freq = 500
# learn_start = 1500
# plot_update_freq = 100
# eval_freq = 500
# eval_episodes = 10
# eps_decay = 1000
# eps_end = 0.1
# hidden_layer = 100
# l1_regularization = 0
# dropout = 0
# is_visdom = True
#
# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')
#
# network = DQN_MLP(input_size, output_size, hidden_layer).to(device)
# network.apply(init_weights)
# target_network = DQN_MLP(input_size, output_size, hidden_layer).to(device)
# target_network.load_state_dict(network.state_dict())
# memory = ReplayBuffer(mem_capacity)
#
# optimizer = optim.Adam(network.parameters(), lr=lr,amsgrad=True)
#
# average_rewards = []
# avg_rew_steps = []
# losses = []
# losses_steps = []
#
# done = True
# for step in range(num_steps):
#
#     if done:
#         if is_chen:
#             state_idx = env.reset()
#             state = torch.zeros([input_size], dtype=torch.float32)
#             state[state_idx] = 1
#         else:
#             state = env.reset()
#             state = np.reshape(state, (1, -1))
#             state = torch.from_numpy(state).to(device)
#             dtype = torch.float32
#             state = Variable(state.type(dtype)).to(device)
#     if is_chen:
#         action = network(state.unsqueeze(0)).cpu().max(1)[1].item()
#     else:
#         if env.startDelay >= 0:
#             # game pre-start
#             action = gym.spaces.np_random.randint(env.action_space.n)
#         else:
#             validActions = env.getValidActions()
#             actionScores = network(state).detach().cpu().numpy().squeeze()
#             actionScores = [actionScores[i] for i in validActions]
#             action = validActions[np.asarray(actionScores).argmax()]
#     eps = max((eps_decay - step + learn_start) / eps_decay, eps_end)
#     if random.random() < eps:
#         if env.startDelay >= 0:
#             # game pre-start
#             action = gym.spaces.np_random.randint(env.action_space.n)
#         else:
#             # rest of the game
#             actions = env.getValidActions()
#             action = actions[gym.spaces.np_random.randint(len(actions))]
#
#     if is_chen:
#         next_state_idx, reward, done, _ = env.step(action)
#         next_state = torch.zeros([input_size], dtype=torch.float32).to(device)
#         next_state[next_state_idx] = 1
#     else:
#         next_state, reward, done, _ = env.step(action)
#         # for the convolutional architecture, we keep it in the original shape
#         next_state = np.reshape(next_state, (1, -1))
#         next_state = torch.from_numpy(next_state).to(device)
#         dtype = torch.float32
#         next_state = Variable(next_state.type(dtype))
#     # after we made a step render it to visualize
#     env.render()
#
#     # update plots
#     if env.done and step % plot_update_freq == 0 and is_visdom:
#         env.updatePlots(is_learn_start=(step > learn_start))
#
#     # Done due to timeout is a non-markovian property. This is an artifact which we would not like to learn from.
#     if not (done and reward < 0):
#         memory.add(state, action, reward, next_state, not done)
#     state = next_state
#
#     if step > learn_start:
#         batch_state, batch_action, batch_reward, batch_next_state, not_done_mask = memory.sample(batch)
#
#         batch_state = torch.stack(batch_state).to(device)
#         batch_next_state = torch.stack(batch_next_state).to(device)
#         batch_action = torch.tensor(batch_action, dtype=torch.int64).unsqueeze(-1).to(device)
#         batch_reward = torch.tensor(batch_reward, dtype=torch.float32).unsqueeze(-1).to(device)
#         not_done_mask = torch.tensor(not_done_mask, dtype=torch.float32).unsqueeze(-1).to(device)
#
#         current_Q = network(batch_state).gather(1, batch_action)
#
#         with torch.no_grad():
#             if double_dqn:
#                 next_state_actions = network(batch_next_state).max(1, keepdim=True)[1]
#                 next_Q = target_network(batch_next_state).gather(1, next_state_actions)
#             else:
#                 next_Q = target_network(batch_next_state).max(1, keepdim=True)[0]
#
#             target_Q = batch_reward + (gamma * next_Q) * not_done_mask
#
#         loss = F.smooth_l1_loss(current_Q, target_Q)
#         # all_params = torch.cat([x.view(-1) for x in model.parameters()])
#         all_params = torch.cat([x.view(-1) for x in network.parameters()]) # TODO - ask chen why it was model and if its really should be network
#         loss += l1_regularization * torch.norm(all_params, 1)
#         loss = torch.clamp(loss, min=-1, max=1)
#
#         if step % plot_update_freq == 0:
#             print('loss is: %f' % loss)
#
#         optimizer.zero_grad()
#         loss.backward()
#         #         for param in network.parameters():
#         #             param.grad.data.clamp_(-1, 1)
#         optimizer.step()
#         losses.append(loss.item())
#         losses_steps.append(step)
#         # # plot losses
#         # plt.figure(4)
#         # plt.plot(losses_steps,losses)
#         # plt.title("Losses")
#         # env.vis.matplot(plt,win=4)
#
#
#
#     if step % target_update_freq == 0:
#         print('target network update')
#         target_network.load_state_dict(network.state_dict())
#
#     if step % eval_freq == 0 and step > learn_start:
#         network.eval()
#         total_reward = 0
#         for eval_ep in range(eval_episodes):
#             if is_chen:
#                 eval_state_idx = eval_env.reset()
#             else:
#                 eval_state = eval_env.reset()
#             while True:
#                 eval_env.render()
#                 if is_chen:
#                     eval_state = torch.zeros([input_size], dtype=torch.float32).to(device)
#                     eval_state[eval_state_idx] = 1
#                     action = network(eval_state.unsqueeze(0)).max(1)[1].item()
#                 else:
#                     eval_state = np.reshape(eval_state, (1, -1))
#                     eval_state = torch.from_numpy(eval_state).to(device)
#                     dtype = torch.float32
#                     eval_state = Variable(eval_state.type(dtype))
#                     # action = network(state).max(1)[1].item()
#                     if eval_env.startDelay >= 0:
#                         # game pre-start
#                         action = gym.spaces.np_random.randint(env.action_space.n)
#                     else:
#                         validActions = eval_env.getValidActions()
#                         actionScores = network(eval_state).detach().cpu().numpy().squeeze()
#                         actionScores = [actionScores[i] for i in validActions]
#                         action = validActions[np.asarray(actionScores).argmax()]
#                 if random.random() < 0.01:
#                     action = random.randrange(output_size)
#                 if is_chen:
#                     eval_state_idx, reward, done, _ = eval_env.step(action)
#                 else:
#                     eval_state, reward, done, _ = eval_env.step(action)
#
#                 total_reward += reward
#                 if done:
#                     break
#         network.train()
#
#         average_reward = total_reward * 1.0 / eval_episodes
#         average_rewards.append(average_reward)
#         avg_rew_steps.append(step)
#         print('Step: ' + str(step) + ' Avg reward: ' + str(average_reward))
#     # if step > learn_start and len(losses) > 0 and len(average_rewards) > 0 and step % 1000 == 0:
#     #     clear_output()
#     #     pl.plot(losses_steps, losses)
#     #     pl.title('Loss')
#     #     pl.show()
#     #     pl.plot(avg_rew_steps, average_rewards)
#     #     pl.title('Reward')
#     #     pl.show()
#
#
# torch.save(network.state_dict(), 'dqn')