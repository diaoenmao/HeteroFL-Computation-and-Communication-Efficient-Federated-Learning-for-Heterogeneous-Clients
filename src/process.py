import os
import itertools
import json
import numpy as np
from utils import save, load, makedir_exist_ok
from config import cfg
import matplotlib.pyplot as plt
import matplotlib.cm as cm

result_path = './output/result'
vis_path = './output/vis'
num_experiments = 1
exp = [str(x) for x in list(range(num_experiments))]
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
model_split_rate = {'a': 1, 'b': 0.5, 'c': 0.25, 'd': 0.125, 'e': 0.0625}
model_split_rate_key = list(model_split_rate.keys())
colors = cm.rainbow(np.linspace(1, 0, len(model_split_rate_key)))
model_color = {model_split_rate_key[i]: colors[i] for i in range(len(model_split_rate_key))}
interp_label_name = ['a-b', 'a-c', 'a-d', 'a-e', 'b-c', 'b-d', 'b-e', 'c-d', 'c-e', 'd-e']


def make_control_list(data_name):
    data_name_dict = {'MNIST': 'conv', 'CIFAR10': 'resnet18', 'WikiText2': 'transformer'}
    model_name = data_name_dict[data_name]
    no_fed_control = [exp, [data_name], ['label'], [model_name], ['0'], ['1'], ['1'], ['none'], ['fix'],
                      combination_mode]
    fed_single_control = [exp, [data_name], ['label'], [model_name], ['1'], ['100'], ['0.1'], ['iid', 'non-iid-2'],
                          ['fix'], combination_mode]
    fed_combination_control = [exp, [data_name], ['label'], [model_name], ['1'], ['100'], ['0.1'], ['iid', 'non-iid-2'],
                               ['dynamic'], combination]
    fed_interp_control = [exp, [data_name], ['label'], [model_name], ['1'], ['100'], ['0.1'], ['iid', 'non-iid-2'],
                          ['fix'], interp]
    local_combination_control = [exp, [data_name], ['label'], [model_name], ['2'], ['100'], ['0.1'], ['iid'], ['fix'],
                                 combination]
    local_interp_control = [exp, [data_name], ['label'], [model_name], ['2'], ['100'], ['0.1'], ['iid'], ['fix'],
                            interp]
    controls_list = [no_fed_control, fed_single_control, fed_combination_control, fed_interp_control]
    return controls_list


def main():
    mnist_control_list = make_control_list('MNIST')
    cifar10_control_list = make_control_list('CIFAR10')
    controls_list = mnist_control_list + cifar10_control_list
    controls = []
    for i in range(len(controls_list)):
        controls.extend(list(itertools.product(*controls_list[i])))
    processed_result_exp, processed_result_history = process_result(controls)
    print(processed_result_exp)
    with open('{}/processed_result_exp.json'.format(result_path), 'w') as fp:
        json.dump(processed_result_exp, fp, indent=2)
    save(processed_result_exp, os.path.join(result_path, 'processed_result_exp.pt'))
    save(processed_result_history, os.path.join(result_path, 'processed_result_history.pt'))
    extracted_processed_result = {}
    extract_processed_result(extracted_processed_result, processed_result_exp, [])
    make_vis(extracted_processed_result)
    return


def process_result(controls):
    processed_result_exp, processed_result_history = {}, {}
    for control in controls:
        model_tag = '_'.join(control)
        extract_result(list(control), model_tag, processed_result_exp, processed_result_history)
    summarize_result(processed_result_exp)
    summarize_result(processed_result_history)
    return processed_result_exp, processed_result_history


def extract_result(control, model_tag, processed_result_exp, processed_result_history):
    if len(control) == 1:
        exp_idx = exp.index(control[0])
        base_result_path_i = os.path.join(result_path, '{}.pt'.format(model_tag))
        if os.path.exists(base_result_path_i):
            if 'params' not in processed_result_exp:
                num_params, num_flops, ratio = make_stats(model_tag)
                processed_result_exp['Params'] = {'exp': [num_params]}
                processed_result_exp['FLOPs'] = {'exp': [num_flops]}
                processed_result_exp['Ratio'] = {'exp': [ratio]}
            base_result = load(base_result_path_i)
            for k in base_result['logger']['test'].mean:
                metric_name = k.split('/')[1]
                if metric_name not in processed_result_exp:
                    processed_result_exp[metric_name] = {'exp': [None for _ in range(num_experiments)]}
                    processed_result_history[metric_name] = {'history': [None for _ in range(num_experiments)]}
                processed_result_exp[metric_name]['exp'][exp_idx] = base_result['logger']['test'].mean[k]
                processed_result_history[metric_name]['history'][exp_idx] = base_result['logger']['train'].history[k]
        else:
            print('Missing {}'.format(base_result_path_i))
    else:
        if control[1] not in processed_result_exp:
            processed_result_exp[control[1]] = {}
            processed_result_history[control[1]] = {}
        extract_result([control[0]] + control[2:], model_tag, processed_result_exp[control[1]],
                       processed_result_history[control[1]])
    return


def summarize_result(processed_result):
    if 'exp' in processed_result:
        pivot = 'exp'
        processed_result[pivot] = np.stack(processed_result[pivot], axis=0)
        processed_result['mean'] = np.mean(processed_result[pivot], axis=0).item()
        processed_result['std'] = np.std(processed_result[pivot], axis=0).item()
        processed_result['max'] = np.max(processed_result[pivot], axis=0).item()
        processed_result['min'] = np.min(processed_result[pivot], axis=0).item()
        processed_result['argmax'] = np.argmax(processed_result[pivot], axis=0).item()
        processed_result['argmin'] = np.argmin(processed_result[pivot], axis=0).item()
        processed_result[pivot] = processed_result[pivot].tolist()
    elif 'history' in processed_result:
        pivot = 'history'
        processed_result[pivot] = np.stack(processed_result[pivot], axis=0)
        processed_result['mean'] = np.mean(processed_result[pivot], axis=0)
        processed_result['std'] = np.std(processed_result[pivot], axis=0)
        processed_result['max'] = np.max(processed_result[pivot], axis=0)
        processed_result['min'] = np.min(processed_result[pivot], axis=0)
        processed_result['argmax'] = np.argmax(processed_result[pivot], axis=0)
        processed_result['argmin'] = np.argmin(processed_result[pivot], axis=0)
        processed_result[pivot] = processed_result[pivot].tolist()
    else:
        for k, v in processed_result.items():
            summarize_result(v)
        return
    return


def extract_processed_result(extracted_processed_result, processed_result, control):
    if 'exp' in processed_result:
        data_name = control[0]
        model_name = control[2]
        control_name = control[-2]
        metric_name = control[-1]
        fig_name = '_'.join([data_name, model_name])
        if fig_name not in extracted_processed_result:
            extracted_processed_result[fig_name] = {}
        if '-' in control_name:
            label_name = '-'.join(['{}'.format(x[0]) for x in list(control_name.split('-'))])
            if label_name not in extracted_processed_result[fig_name]:
                extracted_processed_result[fig_name][label_name] = {'rate': [], 'loss': [], 'acc': []}
            extracted_processed_result[fig_name][label_name][metric_name].append(processed_result['mean'])
        else:
            for label_name in interp_label_name:
                if control_name[0] in label_name:
                    if label_name not in extracted_processed_result[fig_name]:
                        extracted_processed_result[fig_name][label_name] = {'rate': [], 'loss': [], 'acc': []}
                    extracted_processed_result[fig_name][label_name][metric_name].append(processed_result['mean'])
    else:
        for k, v in processed_result.items():
            extract_processed_result(extracted_processed_result, v, control + [k])
    return


def make_vis(extracted_processed_result):
    fig = {}
    for fig_name in extracted_processed_result:
        for label_name in extracted_processed_result[fig_name]:
            fig[fig_name] = plt.figure(fig_name)
            rate = np.array(extracted_processed_result[fig_name][label_name]['rate'])
            loss = np.array(extracted_processed_result[fig_name][label_name]['loss'])
            acc = np.array(extracted_processed_result[fig_name][label_name]['acc'])
            sorted_idx = np.argsort(rate)
            rate, loss, acc = rate[sorted_idx], loss[sorted_idx], acc[sorted_idx]
            plt.plot(rate, acc, '^--', label=label_name)
    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        plt.xlabel('rate')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.legend()
        fig_path = '{}/{}.{}'.format(vis_path, fig_name, cfg['save_format'])
        makedir_exist_ok(vis_path)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return


def make_stats(model_tag):
    model_tag_list = model_tag.split('_')
    data_name = model_tag_list[1]
    model_name = model_tag_list[3]
    model_mode = model_tag_list[-1]
    model_mode_list = model_mode.split('-')
    global_model_mode = model_mode_list[0][0]
    stats_result_path = os.path.join(result_path, '{}_{}_{}.pt'.format(data_name, model_name, global_model_mode))
    stats_result = load(stats_result_path)
    global_num_params = stats_result['num_params']
    all_global_num_params = 0
    num_params = 0
    num_flops = 0
    frac = 0
    for i in range(len(model_mode_list)):
        model_mode_i = model_mode_list[i][0]
        frac_i = int(model_mode_list[i][1])
        stats_result_path_i = os.path.join(result_path, '{}_{}_{}.pt'.format(data_name, model_name, model_mode_i))
        stats_result = load(stats_result_path_i)
        num_params += stats_result['num_params'] * frac_i
        num_flops += stats_result['num_flops'] * frac_i
        all_global_num_params += global_num_params * frac_i
        frac += frac_i
    ratio = num_params / all_global_num_params
    num_params /= frac
    num_flops /= frac
    return num_params, num_flops, ratio


if __name__ == '__main__':
    main()