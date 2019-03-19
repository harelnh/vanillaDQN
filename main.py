import train
import train_atary
import os
import copy

base_dir = os.path.abspath('results_2_fruit')
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

is_run_atary_drqn = True
is_run_drqn = False
is_run_vanilla_dqn = True
is_run_grid_search =  True
is_run_best_results = False


# these are our default params.
kwargs = {
    'grid_dim': 3,
    'num_of_obj': 1,
    'mem_capacity': 100000,
    'batch' : 128,
    'lr' : 0.001,
    'double_dqn' : False,
    'gamma' : 0.99,
    'num_steps' : 300000,
    'target_update_freq': 500,
    'learn_start' : 400,
    'plot_update_freq' : 100,
    'eval_freq' : 500,
    'eval_episodes' : 3,
    'eps_decay' : 1000,
    'eps_end' : 0.1,
    'inner_linear_dim' : 100,
    'l1_regularization': 0,
    'dropout' : 0,
    'maxSteps' : 30,
    'is_visdom' : True,
    'output_path' : base_dir,
    'write_mode' : 'w',
}


if is_run_atary_drqn:
    output_dir = os.path.abspath('atary_lstm')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_path = output_dir + '/log'
    cur_kwargs = copy.deepcopy(kwargs)
    cur_kwargs['output_path'] = output_path
    cur_kwargs['hidden_dim'] = 128
    cur_kwargs['lstm_layers'] = 10
    cur_kwargs['batch'] = 1
    cur_kwargs['traj_len'] = 100
    cur_kwargs['is_visdom'] = True

    result = train_atary.train_atary_lstm(**cur_kwargs)

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

    result = train.train_drqn_sequential(**cur_kwargs)

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
            avg_rewards.append(train.train_vannila_dqn(**cur_kwargs))
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
            train.train_vannila_dqn(**cur_kwargs)

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
            train.train_vannila_dqn(**cur_kwargs)

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
            train.train_vannila_dqn(**cur_kwargs)

