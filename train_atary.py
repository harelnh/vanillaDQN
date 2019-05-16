import math
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
from DRQN_atary import DRQN_atary, ReplayBuffer, init_weights
from GridWorldSimon import gameEnv
from torch.autograd import Variable
import matplotlib.pyplot as plt
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch

random.seed(3)


def train_atary_lstm(**kwargs):

    random.seed(3)

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
    inner_linear_dim = kwargs['inner_linear_dim']
    hidden_dim = kwargs['hidden_dim']
    lstm_layers = kwargs['lstm_layers']
    l1_regularization = kwargs['l1_regularization']
    dropout = kwargs['dropout']
    is_visdom = kwargs['is_visdom']
    write_mode = kwargs['write_mode']
    traj_len = kwargs['traj_len']
    is_rnn = kwargs['is_rnn']

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    env_id = "PongNoFrameskip-v4"
    env = make_atari(env_id)
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)
    eval_env = make_atari(env_id)
    eval_env = wrap_deepmind(eval_env)
    eval_env = wrap_pytorch(eval_env)


    # env = gameEnv(size=grid_dim, startDelay=num_of_obj, maxSteps=maxSteps - 2)
    # eval_env = gameEnv(size=grid_dim, startDelay=num_of_obj, maxSteps=maxSteps - 2)
    # input_size = env.observation_space.n
    input_size = env.observation_space.shape
    output_size = env.action_space.n

    Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done','pad_mask'))

    def pad_episode(episode_transitions):

        zero_transition = Transition(np.zeros(episode_transitions[0][0].shape),
                                     0, 0, np.zeros(episode_transitions[0][0].shape), 0, 0)

        for i in range(traj_len - len(episode_transitions)):
            episode_transitions.append(zero_transition)
        return episode_transitions


    f = open(kwargs['output_path'], write_mode)

    network = DRQN_atary(input_size, output_size, inner_linear_dim,hidden_dim,lstm_layers,batch, traj_len, seed=3, device = device,is_rnn = is_rnn).to(device)
    target_network = DRQN_atary(input_size, output_size, inner_linear_dim, hidden_dim,lstm_layers,batch,traj_len, seed=3,device = device,is_rnn = is_rnn).to(device)
    target_network.load_state_dict(network.state_dict())

    # network.load_state_dict(torch.load('drqn_-20.45854483924511'))
    # target_network.load_state_dict(torch.load('drqn_-20.45854483924511'))

    memory = ReplayBuffer(mem_capacity, batch)

    optimizer = optim.Adam(network.parameters(), lr=lr)

    average_rewards = []
    avg_rew_steps = []
    losses = []
    losses_steps = []
    episode_transitions = []
    done = True
    traj_steps_cnt = 0
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 30000
    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1. * frame_idx / epsilon_decay)

    for step in range(num_steps):


        if done or traj_steps_cnt % traj_len == 0:
            traj_steps_cnt = 0
            if len(episode_transitions) > 0:
                episode_transitions = pad_episode(episode_transitions)
                memory.add_episode(episode_transitions)
            episode_transitions = []
            if done:
                state = env.reset()
                network.hidden = network.init_hidden()


        traj_steps_cnt += 1
        # old epsilon
        # eps = max((eps_decay - step + learn_start) / eps_decay, eps_end)
        # new epsilon
        eps = epsilon_by_frame(step)

        if random.random() > eps:
            q_value = network(Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0).to(device), volatile=True))
            q_value = q_value.view(-1,output_size).cpu().detach().numpy()
            action = np.argmax(q_value)
        else:
            action = random.randrange(env.action_space.n)


        next_state, reward, done, _ = env.step(action)

        # after we made a step render it to visualize
        if is_visdom:
            env.render()

        # update plots
        # if env.done and step % plot_update_freq == 0 and is_visdom:
        #     env.updatePlots(is_learn_start=(step > learn_start))

        # Done due to timeout is a non-markovian property. This is an artifact which we would not like to learn from.
        # if not (done and reward < 0):
            # memory.add(state, action, reward, next_state, not done)
        episode_transitions.append(Transition(state, action, reward, next_state, not done, 1)) # Todo - done or not done

        state = next_state

        # save the current hidden vector to restore it after training step
        so_far_hidden = network.clone_hidden()

        # train part
        if step > learn_start:
            # TODO - is it better to save the hidden vec too in the beggining of each traj, or maybe it's wrong since the weights are changing
            network.batch_hidden = network.init_batch_hidden()
            target_network.batch_hidden = target_network.init_batch_hidden()
            optimizer.zero_grad()


            batch_state, batch_action, batch_reward, batch_next_state, not_done_mask, is_pad_mask = memory.sample_episode()


            batch_state = Variable(torch.FloatTensor(np.float32(batch_state)).to(device))
            batch_next_state = Variable(torch.FloatTensor(np.float32(batch_next_state)).to(device), volatile=True)
            batch_action = torch.tensor(batch_action, dtype=torch.int64).view(batch * traj_len,-1).to(device)
            batch_reward = torch.tensor(batch_reward, dtype=torch.float32).view(batch * traj_len,-1).to(device)
            not_done_mask = torch.tensor(not_done_mask, dtype=torch.float32).view(batch * traj_len,-1).to(device)
            is_pad_mask = torch.tensor(is_pad_mask, dtype=torch.float32).view(batch * traj_len,-1).to(device)

            # current_Q = network.forward_batch(batch_state).view(-1,4).gather(1, batch_action) * is_pad_mask
            current_Q = network.forward(batch_state).view(-1,output_size).gather(1, batch_action) * is_pad_mask
            # current_Q = network(batch_state).view(batch,-1).gather(1, batch_action) * is_pad_mask


            with torch.no_grad():
                if double_dqn:
                    next_state_actions = network(batch_next_state).max(1, keepdim=True)[1]
                    next_Q = target_network(batch_next_state).gather(1, next_state_actions)
                else:
                    next_Q = target_network.forward(batch_next_state).view(-1,output_size).max(1, keepdim=True)[0]

                target_Q = batch_reward + (gamma * next_Q) * not_done_mask * is_pad_mask

            # loss = F.smooth_l1_loss(current_Q, target_Q)
            loss = (current_Q - target_Q).pow(2).mean()
            # all_params = torch.cat([x.view(-1) for x in model.parameters()])
            all_params = torch.cat([x.view(-1) for x in
                                    network.parameters()])
            # loss += l1_regularization * torch.norm(all_params, 1)
            #TODO do we want to clamp like this, maybe the intersting info is above abs(1) so we need to use tanh or etc.
            # loss = torch.clamp(loss, min=-1, max=1)

            if step % plot_update_freq == 0:
                print('loss is: %f' % loss)

            loss.backward()
            # found as helpful to limit max grad values
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

        # after training session we restore the hidden vector values
        network.hidden = so_far_hidden

        if step % target_update_freq == 0:
            # print('target network update')
            target_network.load_state_dict(network.state_dict())
        # TODO - adapt to atary code
        if step % eval_freq == 0 and step > learn_start:
            network.eval()
            # save the current hidden vector to restore it after training step
            so_far_hidden = network.clone_hidden()

            total_reward = 0
            for eval_ep in range(eval_episodes):

                network.hidden = network.init_hidden()
                eval_state = eval_env.reset()
                while True:
                    # if is_visdom:
                    eval_env.render()

                    # action = network(state).max(1)[1].item()

                    q_value = network(
                        Variable(torch.FloatTensor(np.float32(eval_state)).unsqueeze(0).to(device), volatile=True))
                    q_value = q_value.view(-1, output_size).cpu().detach().numpy()
                    action = np.argmax(q_value)

                    if random.random() < 0.01:
                        action = random.randrange(output_size)

                    eval_state, reward, done, _ = eval_env.step(action)

                    total_reward += reward
                    if done:
                        break
            network.train()

            # after evaluation session we restore the hidden vector values
            network.hidden = so_far_hidden

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



    tot_avg_reward = sum(average_rewards) / (float(len(average_rewards)) + 0.0000000001)
    print('Run average reward: ' + str(tot_avg_reward))
    f.write('Run average reward: ' + str(tot_avg_reward) + '\n')
    f.close()
    torch.save(network.state_dict(), 'drqn_' + str(tot_avg_reward))
    return tot_avg_reward

