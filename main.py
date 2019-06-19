import train_gridworld
import train_atari
import os
import copy
import datetime


base_dir = os.path.abspath('results_2_fruit')
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

is_run_atary_drqn = True
is_run_drqn = True
is_run_vanilla_dqn = False
is_run_grid_search =  False
is_run_best_results = False


# these are our default params.
kwargs = {
    'mem_capacity': 100000,
    'grid_dim': 3,
    'num_of_obj': 1,
    'batch' : 32,
    'lr' : 0.00001,
    'double_dqn' : False,
    'gamma' : 0.99,
    'num_steps' : 2000000,
    'target_update_freq': 1,
    'learn_start' : 10000,
    'plot_update_freq' : 1000,
    'eval_freq' : 7000,
    'eval_episodes' : 3,
    'eps_decay' : 30000,
    'eps_end' : 0.01,
    'inner_linear_dim' : 512,
    'l1_regularization': 0,
    'dropout' : 0,
    'maxSteps' : 30,
    'is_visdom' : True,
    'output_path' : base_dir,
    'write_mode' : 'w',
    'is_rnn' : False,
    'traj_len': 10,
    'hidden_dim': 256,
    'lstm_layers': 1,
    'flickering_p': 0,
}

if is_run_atary_drqn:
    for is_rnn in [False]:
        for flickering_p in [0,0.1]:
            for traj_len in [10]:
                kwargs['is_rnn'] = is_rnn
                kwargs['traj_len'] = traj_len
                kwargs['flickering_p'] = flickering_p
                print("start training drqn. lr: {:f} batch size: {:f} trajectory length: {:f} flickering p {:f} is_rnn: {:s}".format(kwargs['lr'],
                                                                 kwargs['batch'], kwargs['traj_len'], kwargs['flickering_p'], str(kwargs['is_rnn'])))
                print(datetime.datetime.now())
                output_dir = os.path.abspath('atary_lstm')
                kwargs['output_path'] = output_dir + "/lr_{:f}_batch_size:_{:f}_trajectory_length:_{:f}_flickering_p_{:f}_is_rnn:_{:s}".format(kwargs['lr'],
                                                                 kwargs['batch'], kwargs['traj_len'], kwargs['flickering_p'], str(kwargs['is_rnn']))
                result = train_atari.train_atari_lstm(**kwargs)




is_greed_search = True
if is_greed_search:

    if is_run_atary_drqn:
        output_dir = os.path.abspath('atary_lstm')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_path = output_dir + '/log'
        cur_kwargs = copy.deepcopy(kwargs)
        cur_kwargs['output_path'] = output_path
        cur_kwargs['hidden_dim'] = 128
        cur_kwargs['lstm_layers'] = 10
        cur_kwargs['batch'] = 32
        # lr_range = [0.00001,0.00025]
        lr_range = [0.00001, 0.00025]
        traj_len_range = [1,10]
        target_update_freq_range = [1,100,500]
        batch_size_range = [32]
        for lr in lr_range:
            for traj_len in traj_len_range:
                for batch in batch_size_range:
                    for update_freq in target_update_freq_range:
                        print ("start training drqn. lr: {:f} batch size: {:f} trajectory length: {:f} update frequency {:f}".format(lr,batch,traj_len,update_freq))
                        cur_kwargs['lr'] = lr
                        cur_kwargs['batch'] = batch
                        cur_kwargs['traj_len'] = traj_len
                        cur_kwargs['target_update_freq'] = update_freq
                        cur_kwargs['output_path'] = output_dir + '/lr_' + str(lr) + '_batch_' + str(batch) + '_traj_len_' + str(traj_len) + '_update_freq_' + str(update_freq)
                        result = train_atari.train_atari_lstm(**cur_kwargs)

    if is_run_drqn:
        output_dir = os.path.abspath('seqential_sampling')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_path = output_dir + '/log'
        cur_kwargs = copy.deepcopy(kwargs)
        cur_kwargs['output_path'] = output_path
        cur_kwargs['hidden_dim'] = 128
        cur_kwargs['lstm_layers'] = 10
        cur_kwargs['batch'] = 1
        cur_kwargs['lr'] = 0.00001

        result = train_gridworld.train_drqn_sequential(**cur_kwargs)

    if is_run_vanilla_dqn:

        if is_run_best_results:
            cur_kwargs = copy.deepcopy(kwargs)
            lr = 0.001
            batch = 128
            target_update_freq = 1500
            dir_name = '/best_config'
            output_path =  (base_dir + dir_name + '/' + 'batch_'
            + str(batch) + '_' + '_lr_' + str(lr) + '_target_update_freq_' + str(target_update_freq))
            cur_kwargs['lr'] = lr
            cur_kwargs['batch'] = batch
            cur_kwargs['target_update_freq'] = target_update_freq
            cur_kwargs['output_path'] = output_path

            if not os.path.exists(base_dir + dir_name):
                os.mkdir(base_dir + dir_name)
            avg_rewards = []
            iter_num = 10
            for iter in range(iter_num):
                avg_rewards.append(train_gridworld.train_vannila_dqn(**cur_kwargs))
                cur_kwargs['write_mode'] = 'a'
            f = open(output_path,'a')
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
                cur_kwargs['output_path'] = base_dir + dir_name + '/' + 'lr_' + str(lr)
                train_gridworld.train_vannila_dqn(**cur_kwargs)

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
                cur_kwargs['output_path'] = base_dir + dir_name + '/' + 'batch_' + str(batch_size)
                train_gridworld.train_vannila_dqn(**cur_kwargs)

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
                cur_kwargs['output_path'] = base_dir + dir_name + '/' + 'target_update_freq_' + str(update_freq)
                train_gridworld.train_vannila_dqn(**cur_kwargs)

