import argparse
import itertools

parser = argparse.ArgumentParser(description='Config')
parser.add_argument('--run', default='train', type=str)
parser.add_argument('--file', default='classifier', type=str)
parser.add_argument('--model', default=None, type=str)
parser.add_argument('--fed', default=1, type=int)
parser.add_argument('--num_gpu', default=4, type=int)
parser.add_argument('--world_size', default=1, type=int)
parser.add_argument('--round', default=4, type=int)
parser.add_argument('--experiment_step', default=1, type=int)
parser.add_argument('--num_experiments', default=1, type=int)
parser.add_argument('--resume_mode', default=0, type=int)
parser.add_argument('--data_split_mode', default='iid', type=str)
args = vars(parser.parse_args())


def main():
    run = args['run']
    file = args['file']
    model = args['model']
    fed = args['fed']
    num_gpu = args['num_gpu']
    world_size = args['world_size']
    round = args['round']
    experiment_step = args['experiment_step']
    num_experiments = args['num_experiments']
    resume_mode = args['resume_mode']
    data_split_mode = args['data_split_mode']
    gpu_ids = [','.join(str(i) for i in list(range(x, x + world_size))) for x in list(range(0, num_gpu, world_size))]
    filename = '{}_{}'.format(run, model)
    if run == 'train' and fed:
        script_name = [['{}_{}_fed.py'.format(run, file)]]
    else:
        script_name = [['{}_{}.py'.format(run, file)]]
    if model in ['conv']:
        data_names = [['MNIST']]
    elif model in ['resnet18']:
        data_names = [['CIFAR10']]
    else:
        raise ValueError('Not valid model')
    model_names = [[model]]
    init_seeds = [list(range(0, num_experiments, experiment_step))]
    world_size = [[world_size]]
    num_experiments = [[experiment_step]]
    resume_mode = [[resume_mode]]
    model_split_mode = ['a', 'b', 'c', 'd', 'e']
    combination_mode = [x + '1' for x in model_split_mode]
    combination = []
    for i in range(1, len(combination_mode) + 1):
        combination_mode_i = ['-'.join(list(x)) for x in itertools.combinations(combination_mode, i)]
        combination.extend(combination_mode_i)
    combination = combination[5:]
    interp = []
    for i in range(1, 10):
        for j in range(len(model_split_mode)):
            for k in range(j + 1, len(model_split_mode)):
                interp += ['{}{}-'.format(model_split_mode[j], i) + '{}{}'.format(model_split_mode[k], 10 - i)]
    if fed == 0:
        control_name = [[['SGD'], ['1'], ['1'], ['none'], ['fix'], model_split_mode]]
    elif fed == 1:
        control_name_single = [['SGD'], ['100'], ['0.1'], [data_split_mode], ['fix'], ['a'],
                               ['a1', 'b1', 'c1', 'd1', 'e1']]
        control_name_combination = [['SGD'], ['100'], ['0.1'], [data_split_mode], ['fix', 'dynamic'], ['a'],
                                    combination]
        control_name_interp = [['SGD'], ['100'], ['0.1'], [data_split_mode], ['fix', 'dynamic'], ['a'], interp]
        control_name_full = [['SGD'], ['100'], ['0.1'], [data_split_mode], ['fix'],
                             ['b_a1', 'c_a1', 'd_a1', 'e_a1']]
        control_name = [control_name_single, control_name_combination, control_name_interp, control_name_full]
    else:
        raise ValueError('Not valid fed')
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    control_names = [control_names]
    controls = script_name + data_names + model_names + init_seeds + world_size + num_experiments + resume_mode + \
               control_names
    controls = list(itertools.product(*controls))
    s = '#!/bin/bash\n'
    k = 0
    for i in range(len(controls)):
        controls[i] = list(controls[i])
        s = s + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --data_name {} --model_name {} --init_seed {} ' \
                '--world_size {} --num_experiments {} --resume_mode {} --control_name {}&\n'.format(
            gpu_ids[k % len(gpu_ids)], *controls[i])
        if k % round == round - 1:
            s = s[:-2] + '\nwait\n'
        k = k + 1
    if s[-5:-1] != 'wait':
        s = s + 'wait\n'
    print(s)
    run_file = open('./{}.sh'.format(filename), 'w')
    run_file.write(s)
    run_file.close()
    return


if __name__ == '__main__':
    main()