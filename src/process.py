import os
import itertools
import json
import numpy as np
from utils import save, load, makedir_exist_ok
import matplotlib.pyplot as plt
import matplotlib.cm as cm

result_path = './output/result'
vis_path = './output/vis'
num_experiments = 1
exp = [str(x) for x in list(range(num_experiments))]
model_split_mode = ['a', 'b', 'c', 'd', 'e']
model_split = []
for i in range(1, len(model_split_mode) + 1):
    model_split_i = [''.join(list(x)) for x in itertools.combinations(model_split_mode, i)]
    model_split.extend(model_split_i)
model_split_rate = {'a': 1, 'b': 0.5, 'c': 0.25, 'd': 0.125, 'e': 0.0625}
model_split_rate_key = list(model_split_rate.keys())
model_color = {model_split_rate_key[i]: i for i in range(len(model_split_rate_key))}
colors = cm.rainbow(np.linspace(1, 0, len(model_split_rate_key)))
alpha = 0.75
fig_format = 'png'


def main():
    info = {'controls': [exp, ['MNIST', 'CIFAR10'], ['label'], ['conv'], ['Adam'],
                         ['100'], ['0.1'], ['iid'], model_split], 'metrics': ['test/Accuracy']}
    processed_result = process_result(info)
    with open('{}/summarized.json'.format(result_path), 'w') as fp:
        json.dump(processed_result, fp, indent=2)
    save(processed_result, os.path.join(result_path, 'processed_result.pt'))
    make_vis(processed_result, info['metrics'])
    return


def process_result(info):
    metrics = info['metrics']
    controls = list(itertools.product(*info['controls']))
    processed_result = {}
    for control in controls:
        model_tag = '_'.join(control)
        result_path_i = os.path.join(result_path, '{}.pt'.format(model_tag))
        if os.path.exists(result_path_i):
            result = load(result_path_i)
            extract_result(list(control), processed_result, result, metrics)
    summarize_result([], processed_result, metrics)
    return processed_result


def extract_result(control, processed_result, result, metrics):
    if len(control) == 1:
        for metric in metrics:
            if metric not in processed_result:
                processed_result[metric] = {'exp': np.zeros(num_experiments)}
            exp_idx = exp.index(control[0])
            processed_result[metric]['exp'][exp_idx] = result['logger'].mean[metric]
    else:
        if control[1] not in processed_result:
            processed_result[control[1]] = {}
        extract_result([control[0]] + control[2:], processed_result[control[1]], result, metrics)
    return


def summarize_result(controls, processed_result, metrics):
    if metrics[0] in processed_result:
        for metric in metrics:
            processed_result[metric]['mean'] = np.mean(processed_result[metric]['exp']).item()
            processed_result[metric]['std'] = np.std(processed_result[metric]['exp']).item()
            processed_result[metric]['max'] = np.max(processed_result[metric]['exp']).item()
            processed_result[metric]['min'] = np.min(processed_result[metric]['exp']).item()
            argmax = np.argmax(processed_result[metric]['exp']).item()
            argmin = np.argmin(processed_result[metric]['exp']).item()
            processed_result[metric]['argmax'] = '_'.join([str(argmax)] + controls)
            processed_result[metric]['argmin'] = '_'.join([str(argmin)] + controls)
            processed_result[metric]['exp'] = processed_result[metric]['exp'].tolist()
    else:
        for k, v in processed_result.items():
            summarize_result(controls + [k], v, metrics)
    return


def make_vis(processed_result, metrics):
    fig = {}
    vis_result(fig, [], processed_result, metrics)
    for k, v in fig.items():
        metric_name = k.split('_')[-1].split('/')
        save_fig_name = '_'.join(k.split('_')[:-1] + ['_'.join(metric_name)])
        fig[k] = plt.figure(k)
        plt.xlabel('rate')
        plt.ylabel(metric_name[-1])
        plt.grid()
        fig_path = '{}/{}.{}'.format(vis_path, save_fig_name, fig_format)
        makedir_exist_ok(vis_path)
        fig[k].savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(k)
    return


def vis_result(fig, controls, processed_result, metrics):
    if metrics[0] in processed_result:
        for metric in metrics:
            control_name = controls[-1]
            x = control_name_to_rate(control_name)
            y_mean = processed_result[metric]['mean']
            y_std = processed_result[metric]['std']
            area = y_std + 100
            color = colors[model_color[control_name[0]]]
            fig_name = '_'.join(controls[:-1] + [metric])
            fig[fig_name] = plt.figure(fig_name)
            plt.scatter(x, y_mean, s=area, c=color.reshape(1, -1), alpha=alpha)
            plt.text(x, y_mean, control_name, fontsize=5)
    else:
        for k, v in processed_result.items():
            vis_result(fig, controls + [k], v, metrics)
    return


def control_name_to_rate(control_name):
    rate = 0
    for c in control_name:
        rate += model_split_rate[c] ** 2
    rate /= len(control_name)
    return rate


if __name__ == '__main__':
    main()