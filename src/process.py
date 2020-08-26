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


def main():
    no_fed_control = [exp, ['MNIST', 'CIFAR10'], ['label'], ['conv', 'resnet18'], ['SGD'], ['1'], ['1'],
                      ['none'], combination_mode]
    combination_control = [exp, ['MNIST', 'CIFAR10'], ['label'], ['conv', 'resnet18'], ['SGD'], ['100'], ['0.1'],
                           ['iid'], combination]
    interp_control = [exp, ['MNIST', 'CIFAR10'], ['label'], ['conv', 'resnet18'], ['SGD'], ['100'], ['0.1'],
                      ['iid'], interp + combination_mode]
    controls_list = [interp_control]
    controls = []
    for i in range(len(controls_list)):
        controls.extend(list(itertools.product(*controls_list[i])))
    processed_result = process_result(controls)
    with open('{}/processed_result.json'.format(result_path), 'w') as fp:
        json.dump(processed_result, fp, indent=2)
    save(processed_result, os.path.join(result_path, 'processed_result.pt'))
    extracted_processed_result = {}
    extract_processed_result(extracted_processed_result, [], processed_result)
    make_vis(extracted_processed_result)
    return


def process_result(controls):
    processed_result = {}
    for control in controls:
        model_tag = '_'.join(control)
        extract_result(list(control), processed_result, model_tag)
    summarize_result(processed_result)
    return processed_result


def extract_result(control, processed_result, model_tag):
    if len(control) == 1:
        exp_idx = exp.index(control[0])
        base_result_path_i = os.path.join(result_path, '{}.pt'.format(model_tag))
        if os.path.exists(base_result_path_i):
            if 'rate' not in processed_result:
                processed_result['rate'] = {'exp': [control_name_to_rate(model_tag.split('_')[-1])]}
            if 'loss' not in processed_result:
                processed_result['loss'] = {'exp': [None for _ in range(num_experiments)]}
            if 'acc' not in processed_result:
                processed_result['acc'] = {'exp': [None for _ in range(num_experiments)]}
            base_result = load(base_result_path_i)
            processed_result['loss']['exp'][exp_idx] = base_result['logger'].mean['test/Loss']
            processed_result['acc']['exp'][exp_idx] = base_result['logger'].mean['test/Accuracy']
    else:
        if control[1] not in processed_result:
            processed_result[control[1]] = {}
        extract_result([control[0]] + control[2:], processed_result[control[1]], model_tag)
    return


def summarize_result(processed_result):
    if 'exp' in processed_result:
        processed_result['exp'] = np.stack(processed_result['exp'], axis=0)
        processed_result['mean'] = np.mean(processed_result['exp'], axis=0).item()
        processed_result['std'] = np.std(processed_result['exp'], axis=0).item()
        processed_result['max'] = np.max(processed_result['exp'], axis=0).item()
        processed_result['min'] = np.min(processed_result['exp'], axis=0).item()
        processed_result['argmax'] = np.argmax(processed_result['exp'], axis=0).item()
        processed_result['argmin'] = np.argmin(processed_result['exp'], axis=0).item()
        processed_result['exp'] = processed_result['exp'].tolist()
    else:
        for k, v in processed_result.items():
            summarize_result(v)
    return


def extract_processed_result(extracted_processed_result, control, processed_result):
    if 'exp' in processed_result:
        data_name = control[0]
        model_name = control[2]
        optimizer_name = control[3]
        control_name = control[-2]
        metric_name = control[-1]
        fig_name = '_'.join([data_name, model_name, optimizer_name])
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
            extract_processed_result(extracted_processed_result, control + [k], v)
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


def control_name_to_rate(control_name):
    frac = 0
    rate = 0
    control_name_list = control_name.split('-')
    for control_name in control_name_list:
        frac += int(control_name[1])
        rate += model_split_rate[control_name[0]] ** 2 * int(control_name[1])
    rate /= frac
    return rate


if __name__ == '__main__':
    main()