import os
import itertools
import numpy as np
from config import cfg
from utils import load, makedir_exist_ok
import matplotlib.pyplot as plt

model_path = './output/model'
vis_path = './output/vis/lc'
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
    interp_control = [exp, ['MNIST', 'CIFAR10'], ['label'], ['conv', 'resnet18'], ['SGD'], ['100'], ['0.1'],
                      ['iid'], interp + combination_mode]
    controls_list = [interp_control]
    controls = []
    for i in range(len(controls_list)):
        controls.extend(list(itertools.product(*controls_list[i])))
    processed_result = process_result(controls)
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
        model_path_i = os.path.join(model_path, '{}_checkpoint.pt'.format(model_tag))
        if os.path.exists(model_path_i):
            model = load(model_path_i)
            if 'Accuracy' not in processed_result:
                processed_result['Accuracy'] = {'hist': [None for _ in range(num_experiments)]}
            processed_result['Accuracy']['hist'][exp_idx] = np.array(model['logger'].history['test/Accuracy'])
    else:
        if control[1] not in processed_result:
            processed_result[control[1]] = {}
        extract_result([control[0]] + control[2:], processed_result[control[1]], model_tag)
    return


def summarize_result(processed_result):
    if 'hist' in processed_result:
        processed_result['hist'] = np.stack(processed_result['hist'], axis=0)
        processed_result['mean'] = np.mean(processed_result['hist'], axis=0)
        processed_result['std'] = np.std(processed_result['hist'], axis=0)
        processed_result['max'] = np.max(processed_result['hist'], axis=0)
        processed_result['min'] = np.min(processed_result['hist'], axis=0)
        processed_result['argmax'] = np.argmax(processed_result['hist'], axis=0)
        processed_result['argmin'] = np.argmin(processed_result['hist'], axis=0)
        processed_result['hist'] = processed_result['hist'].tolist()
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


def make_vis(processed_result):
    fig = {}
    vis(fig, [], processed_result)
    for k, v in fig.items():
        metric_name = k.split('_')[-1].split('/')
        save_fig_name = '_'.join(k.split('_')[:-1] + ['_'.join(metric_name)])
        fig[k] = plt.figure(k)
        plt.xlabel('iter', fontsize=12)
        plt.ylabel(metric_name[-1], fontsize=12)
        plt.grid()
        plt.legend()
        fig_path = '{}/{}.{}'.format(vis_path, save_fig_name, cfg['save_format'])
        makedir_exist_ok(vis_path)
        fig[k].savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(k)
    return


def vis(fig, control, processed_result):
    if 'hist' in processed_result:
        data_name = control[0]
        model_name = control[2]
        metric = control[-1]
        if 'mc' in model_name:
            color = colors['mc']
            label = model_name[2:]
        else:
            color = colors['c']
            label = model_name[1:]
        model_name = model_name.upper()
        y_mean = processed_result['mean']
        x = np.arange(y_mean.shape[0])
        y_std = processed_result['std']
        fig_name = '{}_{}_{}'.format(data_name, label, metric)
        fig[fig_name] = plt.figure(fig_name)
        plt.errorbar(x, y_mean, yerr=y_std, c=color, label=model_name)
    else:
        for k, v in processed_result.items():
            vis(fig, control + [k], v)
    return


if __name__ == '__main__':
    main()