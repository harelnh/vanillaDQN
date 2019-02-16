
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import gym
import gym.spaces
import numpy as np
from collections import namedtuple
import random
from matplotlib import pyplot as pl
from IPython.display import clear_output
from DQN import DQN_MLP, ReplayBuffer, init_weights
from GridWorldSimon import gameEnv
from torch.autograd import Variable
import matplotlib.pyplot as plt

random.seed(3)

def train_drqn_sequential_updates(**kwargs):
    random.seed(3)
    Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


    grid_dim = kwargs['grid_dim']
    num_of_obj = kwargs['num_of_obj']
    env = gameEnv(size=grid_dim, startDelay=num_of_obj)
    eval_env = gameEnv(size=grid_dim, startDelay=num_of_obj)

    input_size = env.observation_space.n
    output_size = env.action_space.n
    mem_capacity = kwargs['mem_capacity']
    batch = kwargs['batch']
    lr = kwargs['lr']
    double_dqn = kwargs['double_dqn']
    gamma = kwargs['gamma']
    num_steps = kwargs['num_steps']
    target_update_freq = kwargs['target_update_freq']
    learn_start = kwargs['learn_start']
    plot_update_freq = kwargs['plot_update_freq']
    eval_freq = kwargs['eval_freq']
    eval_episodes = kwargs['eval_episodes']
    eps_decay = kwargs['eps_decay']
    eps_end = kwargs['eps_end']
    hidden_layer = kwargs['hidden_layer']
    l1_regularization = kwargs['l1_regularization']
    dropout = kwargs['dropout']
    is_visdom = kwargs['is_visdom']
    write_mode = kwargs['write_mode']

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    f = open(kwargs['output_dir'], write_mode)

    network = DQN_MLP(input_size, output_size, hidden_layer, seed=3).to(device)
    network.apply(init_weights)
    target_network = DQN_MLP(input_size, output_size, hidden_layer, seed=3).to(device)
    target_network.load_state_dict(network.state_dict())
    memory = ReplayBuffer(mem_capacity)

    optimizer = optim.Adam(network.parameters(), lr=lr, amsgrad=True)

    average_rewards = []
    avg_rew_steps = []
    losses = []
    losses_steps = []
    episode_transitions = []
    done = True
    for step in range(num_steps):


        if done:
            memory.add_episode(episode_transitions)
            episode_transitions = []
            state = env.reset()
            state = np.reshape(state, (1, -1))
            state = torch.from_numpy(state).to(device)
            dtype = torch.float32
            state = Variable(state.type(dtype)).to(device)
            if env.startDelay >= 0:
                # game pre-start
                action = random.randint(0, env.action_space.n - 1)
            else:
                validActions = env.getValidActions()
                actionScores = network(state).detach().cpu().numpy().squeeze()
                actionScores = [actionScores[i] for i in validActions]
                action = validActions[np.asarray(actionScores).argmax()]
        eps = max((eps_decay - step + learn_start) / eps_decay, eps_end)
        if random.random() < eps:
            if env.startDelay >= 0:
                # game pre-start
                action = random.randint(0, env.action_space.n - 1)
            else:
                # rest of the game
                actions = env.getValidActions()
                action = actions[random.randint(0, len(actions) - 1)]

        next_state, reward, done, _ = env.step(action)
        # for the convolutional architecture, we keep it in the original shape
        next_state = np.reshape(next_state, (1, -1))
        next_state = torch.from_numpy(next_state).to(device)
        dtype = torch.float32
        next_state = Variable(next_state.type(dtype))
        # after we made a step render it to visualize
        if is_visdom:
            env.render()

        # update plots
        if env.done and step % plot_update_freq == 0 and is_visdom:
            env.updatePlots(is_learn_start=(step > learn_start))

        # Done due to timeout is a non-markovian property. This is an artifact which we would not like to learn from.
        if not (done and reward < 0):
            # memory.add(state, action, reward, next_state, not done)
            episode_transitions.append(Transition(state, action, reward, next_state, not done))

        state = next_state

        if step > learn_start:
            batch_state, batch_action, batch_reward, batch_next_state, not_done_mask = memory.sample(batch)

            batch_state = torch.stack(batch_state).to(device)
            batch_next_state = torch.stack(batch_next_state).to(device)
            batch_action = torch.tensor(batch_action, dtype=torch.int64).unsqueeze(-1).to(device)
            batch_reward = torch.tensor(batch_reward, dtype=torch.float32).unsqueeze(-1).to(device)
            not_done_mask = torch.tensor(not_done_mask, dtype=torch.float32).unsqueeze(-1).to(device)

            current_Q = network(batch_state).gather(1, batch_action)

            

            with torch.no_grad():
                if double_dqn:
                    next_state_actions = network(batch_next_state).max(1, keepdim=True)[1]
                    next_Q = target_network(batch_next_state).gather(1, next_state_actions)
                else:
                    next_Q = target_network(batch_next_state).max(1, keepdim=True)[0]

                target_Q = batch_reward + (gamma * next_Q) * not_done_mask

            loss = F.smooth_l1_loss(current_Q, target_Q)
            # all_params = torch.cat([x.view(-1) for x in model.parameters()])
            all_params = torch.cat([x.view(-1) for x in
                                    network.parameters()])  # TODO - ask chen why it was model and if its really should be network
            loss += l1_regularization * torch.norm(all_params, 1)
            loss = torch.clamp(loss, min=-1, max=1)

            if step % plot_update_freq == 0:
                print('loss is: %f' % loss)

            optimizer.zero_grad()
            loss.backward()
            #         for param in network.parameters():
            #             param.grad.data.clamp_(-1, 1)
            optimizer.step()
            losses.append(loss.item())
            losses_steps.append(step)
            # # plot losses
            # plt.figure(4)
            # plt.plot(losses_steps,losses)
            # plt.title("Losses")
            # env.vis.matplot(plt,win=4)

        if step % target_update_freq == 0:
            # print('target network update')
            target_network.load_state_dict(network.state_dict())

        if step % eval_freq == 0 and step > learn_start:
            network.eval()
            total_reward = 0
            for eval_ep in range(eval_episodes):

                eval_state = eval_env.reset()
                while True:
                    if is_visdom:
                        eval_env.render()
                    eval_state = np.reshape(eval_state, (1, -1))
                    eval_state = torch.from_numpy(eval_state).to(device)
                    dtype = torch.float32
                    eval_state = Variable(eval_state.type(dtype))
                    # action = network(state).max(1)[1].item()
                    if eval_env.startDelay >= 0:
                        # game pre-start
                        action = random.randint(0, env.action_space.n - 1)
                    else:
                        validActions = eval_env.getValidActions()
                        actionScores = network(eval_state).detach().cpu().numpy().squeeze()
                        actionScores = [actionScores[i] for i in validActions]
                        action = validActions[np.asarray(actionScores).argmax()]
                    if random.random() < 0.01:
                        action = random.randrange(output_size)

                    eval_state, reward, done, _ = eval_env.step(action)

                    total_reward += reward
                    if done:
                        break
            network.train()

            average_reward = total_reward * 1.0 / eval_episodes
            average_rewards.append(average_reward)
            avg_rew_steps.append(step)
            print('Step: ' + str(step) + ' Avg reward: ' + str(average_reward))
            f.write('Step: ' + str(step) + ' Avg reward: ' + str(average_reward) + '\n')
        # if step > learn_start and len(losses) > 0 and len(average_rewards) > 0 and step % 1000 == 0:
        #     clear_output()
        #     pl.plot(losses_steps, losses)
        #     pl.title('Loss')
        #     pl.show()
        #     pl.plot(avg_rew_steps, average_rewards)
        #     pl.title('Reward')
        #     pl.show()
    tot_avg_reward = sum(average_rewards) / float(len(average_rewards))
    print('Run average reward: ' + str(tot_avg_reward))
    f.write('Run average reward: ' + str(tot_avg_reward) + '\n')
    f.close()
    torch.save(network.state_dict(), 'dqn')
    return tot_avg_reward

def train_vannila_dqn(**kwargs):

    random.seed(3)
    Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
    # is_chen = True
    #

    grid_dim = kwargs['grid_dim']
    num_of_obj = kwargs['num_of_obj']
    env = gameEnv(size=grid_dim,startDelay=num_of_obj)
    eval_env = gameEnv(size=grid_dim,startDelay=num_of_obj)


    input_size = env.observation_space.n
    output_size = env.action_space.n
    mem_capacity = kwargs['mem_capacity']
    batch = kwargs['batch']
    lr = kwargs['lr']
    double_dqn = kwargs['double_dqn']
    gamma = kwargs['gamma']
    num_steps = kwargs['num_steps']
    target_update_freq = kwargs['target_update_freq']
    learn_start = kwargs['learn_start']
    plot_update_freq = kwargs['plot_update_freq']
    eval_freq = kwargs['eval_freq']
    eval_episodes = kwargs['eval_episodes']
    eps_decay = kwargs['eps_decay']
    eps_end = kwargs['eps_end']
    hidden_layer = kwargs['hidden_layer']
    l1_regularization = kwargs['l1_regularization']
    dropout = kwargs['dropout']
    is_visdom = kwargs['is_visdom']
    write_mode = kwargs['write_mode']

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    f = open(kwargs['output_dir'], write_mode)

    network = DQN_MLP(input_size, output_size, hidden_layer,seed=3).to(device)
    network.apply(init_weights)
    target_network = DQN_MLP(input_size, output_size, hidden_layer, seed=3).to(device)
    target_network.load_state_dict(network.state_dict())
    memory = ReplayBuffer(mem_capacity)

    optimizer = optim.Adam(network.parameters(), lr=lr,amsgrad=True)

    average_rewards = []
    avg_rew_steps = []
    losses = []
    losses_steps = []

    done = True
    for step in range(num_steps):

        if done:
            state = env.reset()
            state = np.reshape(state, (1, -1))
            state = torch.from_numpy(state).to(device)
            dtype = torch.float32
            state = Variable(state.type(dtype)).to(device)
            if env.startDelay >= 0:
                # game pre-start
                action = random.randint(0,env.action_space.n-1)
            else:
                validActions = env.getValidActions()
                actionScores = network(state).detach().cpu().numpy().squeeze()
                actionScores = [actionScores[i] for i in validActions]
                action = validActions[np.asarray(actionScores).argmax()]
        eps = max((eps_decay - step + learn_start) / eps_decay, eps_end)
        if random.random() < eps:
            if env.startDelay >= 0:
                # game pre-start
                action = random.randint(0,env.action_space.n-1)
            else:
                # rest of the game
                actions = env.getValidActions()
                action = actions[random.randint(0,len(actions)-1)]


        next_state, reward, done, _ = env.step(action)
        # for the convolutional architecture, we keep it in the original shape
        next_state = np.reshape(next_state, (1, -1))
        next_state = torch.from_numpy(next_state).to(device)
        dtype = torch.float32
        next_state = Variable(next_state.type(dtype))
        # after we made a step render it to visualize
        if is_visdom:
            env.render()

        # update plots
        if env.done and step % plot_update_freq == 0 and is_visdom:
            env.updatePlots(is_learn_start=(step > learn_start))

        # Done due to timeout is a non-markovian property. This is an artifact which we would not like to learn from.
        if not (done and reward < 0):
            memory.add(state, action, reward, next_state, not done)
        state = next_state

        if step > learn_start:
            batch_state, batch_action, batch_reward, batch_next_state, not_done_mask = memory.sample(batch)

            batch_state = torch.stack(batch_state).to(device)
            batch_next_state = torch.stack(batch_next_state).to(device)
            batch_action = torch.tensor(batch_action, dtype=torch.int64).unsqueeze(-1).to(device)
            batch_reward = torch.tensor(batch_reward, dtype=torch.float32).unsqueeze(-1).to(device)
            not_done_mask = torch.tensor(not_done_mask, dtype=torch.float32).unsqueeze(-1).to(device)

            current_Q = network(batch_state).gather(1, batch_action)

            with torch.no_grad():
                if double_dqn:
                    next_state_actions = network(batch_next_state).max(1, keepdim=True)[1]
                    next_Q = target_network(batch_next_state).gather(1, next_state_actions)
                else:
                    next_Q = target_network(batch_next_state).max(1, keepdim=True)[0]

                target_Q = batch_reward + (gamma * next_Q) * not_done_mask

            loss = F.smooth_l1_loss(current_Q, target_Q)
            # all_params = torch.cat([x.view(-1) for x in model.parameters()])
            all_params = torch.cat([x.view(-1) for x in network.parameters()]) # TODO - ask chen why it was model and if its really should be network
            loss += l1_regularization * torch.norm(all_params, 1)
            loss = torch.clamp(loss, min=-1, max=1)

            if step % plot_update_freq == 0:
                print('loss is: %f' % loss)

            optimizer.zero_grad()
            loss.backward()
            #         for param in network.parameters():
            #             param.grad.data.clamp_(-1, 1)
            optimizer.step()
            losses.append(loss.item())
            losses_steps.append(step)
            # # plot losses
            # plt.figure(4)
            # plt.plot(losses_steps,losses)
            # plt.title("Losses")
            # env.vis.matplot(plt,win=4)



        if step % target_update_freq == 0:
            # print('target network update')
            target_network.load_state_dict(network.state_dict())

        if step % eval_freq == 0 and step > learn_start:
            network.eval()
            total_reward = 0
            for eval_ep in range(eval_episodes):

                eval_state = eval_env.reset()
                while True:
                    if is_visdom:
                        eval_env.render()
                    eval_state = np.reshape(eval_state, (1, -1))
                    eval_state = torch.from_numpy(eval_state).to(device)
                    dtype = torch.float32
                    eval_state = Variable(eval_state.type(dtype))
                    # action = network(state).max(1)[1].item()
                    if eval_env.startDelay >= 0:
                        # game pre-start
                        action = random.randint(0,env.action_space.n-1)
                    else:
                        validActions = eval_env.getValidActions()
                        actionScores = network(eval_state).detach().cpu().numpy().squeeze()
                        actionScores = [actionScores[i] for i in validActions]
                        action = validActions[np.asarray(actionScores).argmax()]
                    if random.random() < 0.01:
                        action = random.randrange(output_size)

                    eval_state, reward, done, _ = eval_env.step(action)

                    total_reward += reward
                    if done:
                        break
            network.train()

            average_reward = total_reward * 1.0 / eval_episodes
            average_rewards.append(average_reward)
            avg_rew_steps.append(step)
            print('Step: ' + str(step) + ' Avg reward: ' + str(average_reward))
            f.write('Step: ' + str(step) + ' Avg reward: ' + str(average_reward) +'\n')
        # if step > learn_start and len(losses) > 0 and len(average_rewards) > 0 and step % 1000 == 0:
        #     clear_output()
        #     pl.plot(losses_steps, losses)
        #     pl.title('Loss')
        #     pl.show()
        #     pl.plot(avg_rew_steps, average_rewards)
        #     pl.title('Reward')
        #     pl.show()
    tot_avg_reward = sum(average_rewards)/float(len(average_rewards))
    print('Run average reward: ' + str(tot_avg_reward))
    f.write('Run average reward: ' + str(tot_avg_reward) +'\n')
    f.close()
    torch.save(network.state_dict(), 'dqn')
    return tot_avg_reward