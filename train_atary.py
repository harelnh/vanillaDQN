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


    grid_dim = kwargs['grid_dim']
    num_of_obj = kwargs['num_of_obj']
    maxSteps = kwargs['maxSteps']
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

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    env_id = "PongNoFrameskip-v4"
    env = make_atari(env_id)
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)
    eval_env = make_atari(env_id)
    eval_env = wrap_deepmind(env)
    eval_env = wrap_pytorch(env)


    # env = gameEnv(size=grid_dim, startDelay=num_of_obj, maxSteps=maxSteps - 2)
    # eval_env = gameEnv(size=grid_dim, startDelay=num_of_obj, maxSteps=maxSteps - 2)
    # input_size = env.observation_space.n
    input_size = env.observation_space.shape
    output_size = env.action_space.n

    Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done','pad_mask'))

    def pad_episode(episode_transitions):

        zero_transition = Transition(torch.zeros(episode_transitions[0][0].shape).to(device),
                                     0, 0, torch.zeros(episode_transitions[0][0].shape).to(device), 0, 0)

        for i in range(traj_len - len(episode_transitions)):
            episode_transitions.append(zero_transition)
        return episode_transitions


    f = open(kwargs['output_path'], write_mode)

    network = DRQN_atary(input_size, output_size, inner_linear_dim,hidden_dim,lstm_layers,traj_len, seed=3, device = device).to(device)
    network.apply(init_weights)
    target_network = DRQN_atary(input_size, output_size, inner_linear_dim, hidden_dim,lstm_layers,traj_len, seed=3,device = device).to(device)
    target_network.load_state_dict(network.state_dict())
    memory = ReplayBuffer(mem_capacity)

    optimizer = optim.Adam(network.parameters(), lr=lr, amsgrad=True)

    average_rewards = []
    avg_rew_steps = []
    losses = []
    losses_steps = []
    episode_transitions = []
    done = True
    traj_steps_cnt = 0

    for step in range(num_steps):


        if done or traj_steps_cnt % traj_len == 0:
            traj_steps_cnt = 0
            if len(episode_transitions) > 0:
                episode_transitions = pad_episode(episode_transitions)
                memory.add_episode(episode_transitions)
            episode_transitions = []
            state = env.reset()
            network.hidden = network.init_hidden()
            # state = np.reshape(state, (1, -1))
            state = torch.from_numpy(state).to(device)
            dtype = torch.float32
            state = Variable(state.type(dtype = torch.float32)).to(device)

        traj_steps_cnt += 1

        eps = max((eps_decay - step + learn_start) / eps_decay, eps_end)
        if random.random() > eps:
            state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), volatile=True)
            q_value = network(state)
            action = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(env.action_space.n)


        next_state, reward, done, _ = env.step(action)
        # for the convolutional architecture, we keep it in the original shape
        # next_state = np.reshape(next_state, (1, -1))
        next_state = torch.from_numpy(next_state).to(device)
        next_state = Variable(next_state.type(dtype = torch.float32))
        # after we made a step render it to visualize
        if is_visdom:
            env.render()

        # update plots
        # if env.done and step % plot_update_freq == 0 and is_visdom:
        #     env.updatePlots(is_learn_start=(step > learn_start))

        # Done due to timeout is a non-markovian property. This is an artifact which we would not like to learn from.
        # if not (done and reward < 0):
            # memory.add(state, action, reward, next_state, not done)
        episode_transitions.append(Transition(state, action, reward, next_state, not done, 1))

        state = next_state

        if step > learn_start:
            network.hidden = network.init_hidden()
            target_network.hidden = target_network.init_hidden()
            optimizer.zero_grad()


            batch_state, batch_action, batch_reward, batch_next_state, not_done_mask, is_pad_mask = memory.sample_episode()

            batch_state = torch.stack(batch_state).to(device)
            batch_next_state = torch.stack(batch_next_state).to(device)
            batch_action = torch.tensor(batch_action, dtype=torch.int64).unsqueeze(-1).to(device)
            batch_reward = torch.tensor(batch_reward, dtype=torch.float32).unsqueeze(-1).to(device)
            not_done_mask = torch.tensor(not_done_mask, dtype=torch.float32).unsqueeze(-1).to(device)
            is_pad_mask = torch.tensor(is_pad_mask, dtype=torch.float32).unsqueeze(-1).to(device)

            current_Q = network.forward_batch(batch_state).view(-1,4).gather(1, batch_action) * is_pad_mask
            # current_Q = network(batch_state).view(batch,-1).gather(1, batch_action) * is_pad_mask



            with torch.no_grad():
                if double_dqn:
                    next_state_actions = network(batch_next_state).max(1, keepdim=True)[1]
                    next_Q = target_network(batch_next_state).gather(1, next_state_actions)
                else:
                    next_Q = target_network.forward_batch(batch_next_state).view(-1,4).max(1, keepdim=True)[0]

                target_Q = batch_reward + (gamma * next_Q) * not_done_mask * is_pad_mask

            loss = F.smooth_l1_loss(current_Q, target_Q)
            # all_params = torch.cat([x.view(-1) for x in model.parameters()])
            all_params = torch.cat([x.view(-1) for x in
                                    network.parameters()])
            loss += l1_regularization * torch.norm(all_params, 1)
            loss = torch.clamp(loss, min=-1, max=1)

            if step % plot_update_freq == 0:
                print('loss is: %f' % loss)

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

                network.hidden = network.init_hidden()
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
                        # validActions = env.getValidActions()
                        # actionScores, hidden = network(state, hidden)
                        # actionScores = actionScores.detach().cpu().numpy().squeeze()
                        # actionScores = [actionScores[i] for i in validActions]
                        # action = validActions[np.asarray(actionScores).argmax()]

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

