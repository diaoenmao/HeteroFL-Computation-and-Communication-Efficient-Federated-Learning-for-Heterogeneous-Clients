import os
import itertools
import json
import numpy as np
import pandas as pd
from utils import save, load, makedir_exist_ok
from config import cfg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict

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
interp_name = ['a-b', 'a-c', 'a-d', 'a-e', 'b-c', 'b-d', 'b-e', 'c-d', 'c-e', 'd-e', 'e']
marker_dict = {'a-b': 'o', 'a-c': 's', 'a-d': 'p', 'a-e': 'P', 'b-c': '*', 'b-d': 'h', 'b-e': 'X', 'c-d': 'D',
               'c-e': '^', 'd-e': 'v', 'e': '>'}
metric_name_dict = {'MNIST': 'Accuracy', 'CIFAR10': 'Accuracy', 'WikiText2': 'Perplexity'}
loc_dict = {'MNIST': 'lower right', 'CIFAR10': 'lower right', 'WikiText2': 'upper right'}
fontsize = 16


def make_control_list(data_name):
    data_name_dict = {'MNIST': 'conv', 'CIFAR10': 'resnet18', 'WikiText2': 'transformer'}
    data_split_mode_dict = {'MNIST': ['iid', 'non-iid-2'], 'CIFAR10': ['iid', 'non-iid-2'], 'WikiText2': ['iid']}
    model_name = data_name_dict[data_name]
    data_split_mode = data_split_mode_dict[data_name]
    no_fed_control = [exp, [data_name], ['label'], [model_name], ['0'], ['1'], ['1'], ['none'], ['fix'],
                      combination_mode, ['bn'], ['1'], ['1']]
    fed_single_control = [exp, [data_name], ['label'], [model_name], ['1'], ['100'], ['0.1'], data_split_mode,
                          ['fix'], combination_mode, ['bn'], ['1'], ['1']]
    fed_combination_control = [exp, [data_name], ['label'], [model_name], ['1'], ['100'], ['0.1'], data_split_mode,
                               ['dynamic'], combination, ['bn'], ['1'], ['1']]
    fed_interp_control = [exp, [data_name], ['label'], [model_name], ['1'], ['100'], ['0.1'], data_split_mode,
                          ['fix'], interp, ['bn'], ['1'], ['1']]
    controls_list = [no_fed_control, fed_single_control, fed_combination_control, fed_interp_control]
    return controls_list


def make_ablation_control_list(data_name):
    global interp_name
    interp_name = ['a-e', 'e']
    data_name_dict = {'MNIST': 'conv', 'CIFAR10': 'resnet18', 'WikiText2': 'transformer'}
    data_split_mode_dict = {'MNIST': ['iid', 'non-iid-2'], 'CIFAR10': ['iid', 'non-iid-2']}
    model_name = data_name_dict[data_name]
    data_split_mode = data_split_mode_dict[data_name]
    combination = ['a1-e1']
    norm_1 = ['bn', 'none']
    norm_2 = ['in', 'ln', 'gn']
    controls_list = []
    for d in data_split_mode:
        if d == 'iid':
            control_name_1 = [exp, [data_name], ['label'], [model_name], ['1'], ['100'], ['0.1'], [d], ['fix'],
                              ['a1', 'e1'], norm_2 + norm_1, ['1'], ['1']]
            control_name_2 = [exp, [data_name], ['label'], [model_name], ['1'], ['100'], ['0.1'], [d], ['dynamic'],
                              combination, norm_2, ['1'], ['1']]
            control_name_3 = [exp, [data_name], ['label'], [model_name], ['1'], ['100'], ['0.1'], [d], ['dynamic'],
                              combination, norm_1, ['0', '1'], ['1']]
            control_name = [control_name_1, control_name_2, control_name_3]
        elif d == 'non-iid-2':
            control_name_1 = [exp, [data_name], ['label'], [model_name], ['1'], ['100'], ['0.1'], [d], ['fix'],
                              ['a1', 'e1'], norm_2, ['1'], ['1']]
            control_name_2 = [exp, [data_name], ['label'], [model_name], ['1'], ['100'], ['0.1'], [d], ['fix'],
                              ['a1', 'e1'], norm_1, ['1'], ['0', '1']]
            control_name_3 = [exp, [data_name], ['label'], [model_name], ['1'], ['100'], ['0.1'], [d], ['dynamic'],
                              combination, norm_2, ['1'], ['1']]
            control_name_4 = [exp, [data_name], ['label'], [model_name], ['1'], ['100'], ['0.1'], [d], ['dynamic'],
                              combination, norm_1, ['0', '1'], ['0', '1']]
            control_name = [control_name_1, control_name_2, control_name_3, control_name_4]
        controls_list = controls_list + control_name
    return controls_list


def main():
    # mnist_control_list = make_ablation_control_list('MNIST')
    # cifar10_control_list = make_ablation_control_list('CIFAR10')
    # controls_list = mnist_control_list + cifar10_control_list
    mnist_control_list = make_control_list('MNIST')
    cifar10_control_list = make_control_list('CIFAR10')
    wikitext2_control_list = make_control_list('WikiText2')
    controls_list = mnist_control_list + cifar10_control_list + wikitext2_control_list
    controls = []
    for i in range(len(controls_list)):
        controls.extend(list(itertools.product(*controls_list[i])))
    processed_result_exp, processed_result_history = process_result(controls)
    with open('{}/processed_result_exp.json'.format(result_path), 'w') as fp:
        json.dump(processed_result_exp, fp, indent=2)
    save(processed_result_exp, os.path.join(result_path, 'processed_result_exp.pt'))
    save(processed_result_history, os.path.join(result_path, 'processed_result_history.pt'))
    extracted_processed_result = {}
    extract_processed_result(extracted_processed_result, processed_result_exp, [])
    df = make_df(extracted_processed_result)
    make_vis(df)
    extracted_processed_result = {}
    extract_processed_result(extracted_processed_result, processed_result_history, [])
    make_learning_curve(extracted_processed_result)
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
                ratio, num_params, num_flops, space = make_stats(model_tag)
                processed_result_exp['Ratio'] = {'exp': [ratio]}
                processed_result_exp['Params'] = {'exp': [num_params]}
                processed_result_exp['FLOPs'] = {'exp': [num_flops]}
                processed_result_exp['Space'] = {'exp': [space]}
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
    if 'exp' in processed_result or 'history' in processed_result:
        exp_name = '_'.join(control[:-1])
        metric_name = control[-1]
        if exp_name not in extracted_processed_result:
            extracted_processed_result[exp_name] = defaultdict()
        extracted_processed_result[exp_name]['{}_mean'.format(metric_name)] = processed_result['mean']
        extracted_processed_result[exp_name]['{}_std'.format(metric_name)] = processed_result['std']
    else:
        for k, v in processed_result.items():
            extract_processed_result(extracted_processed_result, v, control + [k])
    return


def make_df(extracted_processed_result):
    df = defaultdict(list)
    for exp_name in extracted_processed_result:
        control = exp_name.split('_')
        data_split_mode, model_split_mode, control_name = control[-6:-3]
        except_control_name = control[:-4] + control[-3:]
        index_name = [control_name]
        if data_split_mode == 'none':
            df_name = '_'.join(except_control_name)
            df[df_name].append(pd.DataFrame(data=extracted_processed_result[exp_name], index=index_name))
        else:
            if model_split_mode == 'fix' or control_name == 'e1':
                if '-' in control_name:
                    label_name = '-'.join(['{}'.format(x[0]) for x in list(control_name.split('-'))])
                    df_name = '{}_{}'.format('_'.join(except_control_name), label_name)
                    df[df_name].append(pd.DataFrame(data=extracted_processed_result[exp_name], index=index_name))
                else:
                    for label_name in interp_name:
                        if control_name[0] == label_name[0]:
                            df_name = '{}_{}'.format('_'.join(except_control_name), label_name)
                            df[df_name].append(
                                pd.DataFrame(data=extracted_processed_result[exp_name], index=index_name))
            else:
                df_name = '_'.join(except_control_name)
                df[df_name].append(pd.DataFrame(data=extracted_processed_result[exp_name], index=index_name))
    startrow = 0
    writer = pd.ExcelWriter('{}/result.xlsx'.format(result_path), engine='xlsxwriter')
    for df_name in df:
        df[df_name] = pd.concat(df[df_name])
        df[df_name] = df[df_name].sort_values(by=['Params_mean'], ascending=False)
        df[df_name].to_excel(writer, sheet_name='Sheet1', startrow=startrow + 1)
        writer.sheets['Sheet1'].write_string(startrow, 0, df_name)
        startrow = startrow + len(df[df_name].index) + 3
    writer.save()
    return df


def make_vis(df):
    fig = {}
    for df_name in df:
        if 'fix' in df_name and 'none' not in df_name:
            control = df_name.split('_')
            data_name = control[0]
            metric_name = metric_name_dict[data_name]
            label_name = control[-1]
            x = df[df_name]['Params_mean']
            if 'non-iid-2' in df_name:
                fig_name = '{}_{}_local'.format('_'.join(control[:-1]), label_name[0])
                fig[fig_name] = plt.figure(fig_name)
                y = df[df_name]['Local-{}_mean'.format(metric_name)]
                plt.plot(x, y, linestyle='-', marker=marker_dict[label_name], label=label_name)
                plt.legend(loc=loc_dict[data_name], fontsize=fontsize)
                plt.xlabel('Number of Model Parameters', fontsize=fontsize)
                plt.ylabel(metric_name, fontsize=fontsize)
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)
                plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                fig_name = '{}_{}_global'.format('_'.join(control[:-1]), label_name[0])
                fig[fig_name] = plt.figure(fig_name)
                y = df[df_name]['Global-{}_mean'.format(metric_name)]
                plt.plot(x, y, linestyle='-', marker=marker_dict[label_name], label=label_name)
                plt.legend(loc=loc_dict[data_name], fontsize=fontsize)
                plt.xlabel('Number of Model Parameters', fontsize=fontsize)
                plt.ylabel(metric_name, fontsize=fontsize)
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)
                plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            elif 'iid' in df_name:
                fig_name = '{}_{}'.format('_'.join(control[:-1]), label_name[0])
                fig[fig_name] = plt.figure(fig_name)
                y = df[df_name]['Global-{}_mean'.format(metric_name)]
                plt.plot(x, y, linestyle='-', marker=marker_dict[label_name], label=label_name)
                plt.legend(loc=loc_dict[data_name], fontsize=fontsize)
                plt.xlabel('Number of Model Parameters', fontsize=fontsize)
                plt.ylabel(metric_name, fontsize=fontsize)
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)
                plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            else:
                raise ValueError('Not valid df name')
    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        plt.grid()
        fig_path = '{}/{}.{}'.format(vis_path, fig_name, cfg['save_format'])
        makedir_exist_ok(vis_path)
        plt.savefig(fig_path, dpi=500, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return


def make_learning_curve(processed_result):
    ylim_dict = {'iid': {'global': {'MNIST': [95, 100], 'CIFAR10': [50, 100], 'WikiText2': [0, 20]}},
                 'non-iid-2': {'global': {'MNIST': [50, 100], 'CIFAR10': [0, 70]},
                               'local': {'MNIST': [95, 100], 'CIFAR10': [50, 100]}}}
    fig = {}
    for exp_name in processed_result:
        control = exp_name.split('_')
        data_name = control[0]
        metric_name = metric_name_dict[data_name]
        control_name = control[-4]
        if control_name in ['a5-b5', 'a5-c5', 'a5-d5', 'a5-e5', 'a1-b1', 'a1-c1', 'a1-d1', 'a1-e1']:
            if 'non-iid-2' in exp_name:
                y = processed_result[exp_name]['Local-{}_mean'.format(metric_name)]
                x = np.arange(len(y))
                label_name = '-'.join(['{}'.format(x[0]) for x in list(control_name.split('-'))])
                fig_name = '{}_lc_local'.format('_'.join(control[:-4] + control[-3:]))
                fig[fig_name] = plt.figure(fig_name)
                plt.plot(x, y, linestyle='-', label=label_name)
                plt.legend(loc=loc_dict[data_name], fontsize=fontsize)
                plt.xlabel('Communication rounds', fontsize=fontsize)
                plt.ylabel('Test {}'.format(metric_name), fontsize=fontsize)
                plt.ylim(ylim_dict['non-iid-2']['local'][data_name])
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)
                y = processed_result[exp_name]['Global-{}_mean'.format(metric_name)]
                x = np.arange(len(y))
                label_name = '-'.join(['{}'.format(x[0]) for x in list(control_name.split('-'))])
                fig_name = '{}_lc_global'.format('_'.join(control[:-4] + control[-3:]))
                fig[fig_name] = plt.figure(fig_name)
                plt.plot(x, y, linestyle='-', label=label_name)
                plt.legend(loc=loc_dict[data_name], fontsize=fontsize)
                plt.xlabel('Communication rounds', fontsize=fontsize)
                plt.ylabel('Test {}'.format(metric_name), fontsize=fontsize)
                plt.ylim(ylim_dict['non-iid-2']['global'][data_name])
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)
            else:
                y = processed_result[exp_name]['Global-{}_mean'.format(metric_name)]
                x = np.arange(len(y))
                label_name = '-'.join(['{}'.format(x[0]) for x in list(control_name.split('-'))])
                fig_name = '{}_lc_global'.format('_'.join(control[:-4] + control[-3:]))
                fig[fig_name] = plt.figure(fig_name)
                plt.plot(x, y, linestyle='-', label=label_name)
                plt.legend(loc=loc_dict[data_name], fontsize=fontsize)
                plt.xlabel('Communication rounds', fontsize=fontsize)
                plt.ylabel('Test {}'.format(metric_name), fontsize=fontsize)
                plt.ylim(ylim_dict['iid']['global'][data_name])
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)
    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        plt.grid()
        fig_path = '{}/{}.{}'.format(vis_path, fig_name, cfg['save_format'])
        makedir_exist_ok(vis_path)
        plt.savefig(fig_path, dpi=500, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return


def make_stats(model_tag):
    model_tag_list = model_tag.split('_')
    data_name = model_tag_list[1]
    model_name = model_tag_list[3]
    model_mode = model_tag_list[-4]
    model_mode_list = model_mode.split('-')
    global_model_mode = model_mode_list[0][0]
    stats_result_path = os.path.join(result_path, '{}_{}_{}.pt'.format(data_name, model_name, global_model_mode))
    stats_result = load(stats_result_path)
    global_num_params = stats_result['num_params']
    all_global_num_params = 0
    num_params = 0
    num_flops = 0
    space = 0
    frac = 0
    for i in range(len(model_mode_list)):
        model_mode_i = model_mode_list[i][0]
        frac_i = int(model_mode_list[i][1])
        stats_result_path_i = os.path.join(result_path, '{}_{}_{}.pt'.format(data_name, model_name, model_mode_i))
        stats_result = load(stats_result_path_i)
        num_params += stats_result['num_params'] * frac_i
        num_flops += stats_result['num_flops'] * frac_i
        space += stats_result['space'] * frac_i
        all_global_num_params += global_num_params * frac_i
        frac += frac_i
    ratio = num_params / all_global_num_params
    num_params /= frac
    num_flops /= frac
    space /= frac
    return ratio, num_params, num_flops, space


if __name__ == '__main__':
    main()
