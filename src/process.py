import os
import itertools
import json
import numpy as np
from utils import save, load

data_name = ['CIFAR10', 'Omniglot']
result_path = './output/result'
num_Experiments = 12
control_exp = [str(x) for x in list(range(num_Experiments))]


def main():
    extracted, summarized = {}, {}
    for i in range(len(data_name)):
        result_control = {
            'CVAE': {
                'control_names': [control_exp, [data_name[i]], ['label'], ['cvae']], 'metric': 'test/MSE'},
            'MCVAE': {
                'control_names': [control_exp, [data_name[i]], ['label'], ['mcvae'], ['0.5']], 'metric': 'test/MSE'},
            'VQVAE': {
                'control_names': [control_exp, [data_name[i]], ['label'], ['vqvae']], 'metric': 'test/MSE'},
            'CPixelCNN': {
                'control_names': [control_exp, [data_name[i]], ['label'], ['cpixelcnn']], 'metric': 'test/NLL'},
            'MCPixelCNN': {
                'control_names': [control_exp, [data_name[i]], ['label'], ['mcpixelcnn'], ['0.5']],
                'metric': 'test/NLL'},
            'CGLOW': {
                'control_names': [control_exp, [data_name[i]], ['label'], ['cglow']], 'metric': 'test/Loss'},
            'MCGLOW': {
                'control_names': [control_exp, [data_name[i]], ['label'], ['mcglow'], ['0.5']], 'metric': 'test/Loss'},
            'CGAN': {
                'control_names': [control_exp, [data_name[i]], ['label'], ['cgan']], 'metric': None},
            'MCGAN': {
                'control_names': [control_exp, [data_name[i]], ['label'], ['mcgan'], ['0.5']], 'metric': None},
        }
        extracated_i, summarized_i = {}, {}
        for result_name, info in result_control.items():
            extracated_i[result_name], summarized_i[result_name] = extract_result(info)
        extracted[data_name[i]] = extracated_i
        summarized[data_name[i]] = summarized_i
    print(summarized)
    with open('{}/summarized.json'.format(result_path), 'w') as fp:
        json.dump(summarized, fp, indent=4)
    save((extracted, summarized), '{}/processed_result.pt'.format(result_path))
    make_img(summarized)
    return


def extract_result(info):
    control_names = info['control_names']
    metric = info['metric']
    control_names_product = list(itertools.product(*control_names))
    extracted = {'base': np.zeros(len(control_exp)), 'is': np.zeros((len(control_exp), 2)),
                 'fid': np.zeros(len(control_exp)), 'dbi': np.zeros(len(control_exp))}
    for i in range(len(control_names_product)):
        control_name = list(control_names_product[i])
        model_tag = '_'.join(control_name)
        if metric is not None:
            base_result_path_i = '{}/{}.pt'.format(result_path, model_tag)
            if os.path.exists(base_result_path_i):
                result = load(base_result_path_i)
                exp_idx = control_exp.index(control_names_product[i][0])
                extracted['base'][exp_idx] = result['logger'].mean[metric]
            else:
                pass
        is_result_path_i = '{}/is_{}.npy'.format(result_path, model_tag)
        if os.path.exists(is_result_path_i):
            result = np.load(is_result_path_i)
            exp_idx = control_exp.index(control_names_product[i][0])
            extracted['is'][exp_idx] = result
        else:
            pass
        fid_result_path_i = '{}/fid_{}.npy'.format(result_path, model_tag)
        if os.path.exists(fid_result_path_i):
            result = load(fid_result_path_i, mode='numpy')
            exp_idx = control_exp.index(control_names_product[i][0])
            extracted['fid'][exp_idx] = result
        dbi_result_path_i = '{}/dbi_{}.npy'.format(result_path, model_tag)
        if os.path.exists(dbi_result_path_i):
            result = load(dbi_result_path_i, mode='numpy')
            exp_idx = control_exp.index(control_names_product[i][0])
            extracted['dbi'][exp_idx] = result
        else:
            pass
    best_base = np.min(extracted['base']).item()
    best_is = np.max(extracted['is'][:, 0]).item()
    best_fid = np.min(extracted['fid']).item()
    best_dbi = np.min(extracted['dbi']).item()
    best_name_base = '_'.join(control_names_product[np.argmin(extracted['base']).item()])
    best_name_is = '_'.join(control_names_product[np.argmax(extracted['is'][:, 0]).item()])
    best_name_fid = '_'.join(control_names_product[np.argmin(extracted['fid']).item()])
    best_name_dbi = '_'.join(control_names_product[np.argmin(extracted['dbi']).item()])
    summarized = {
        'base': {'mean': np.mean(extracted['base']), 'stderr': np.std(extracted['base']) / np.sqrt(num_Experiments),
                 'best': best_base, 'best_name': best_name_base},
        'is': {'mean': np.mean(extracted['is'], axis=0).tolist(),
               'stderr': (np.std(extracted['is'], axis=0) / np.sqrt(num_Experiments)).tolist(),
               'best': best_is, 'best_name': best_name_is},
        'fid': {'mean': np.mean(extracted['fid']), 'stderr': np.std(extracted['fid']) / np.sqrt(num_Experiments),
                'best': best_fid, 'best_name': best_name_fid},
        'dbi': {'mean': np.mean(extracted['dbi']), 'stderr': np.std(extracted['dbi']) / np.sqrt(num_Experiments),
                'best': best_dbi, 'best_name': best_name_dbi}}

    return extracted, summarized


def make_img(summarized):
    round = 4
    num_gpu = 1
    gpu_ids = [str(x) for x in list(range(num_gpu))]
    filenames = ['generate', 'transit', 'create']
    save_per_mode = {'generate': {'CIFAR10': 10, 'Omniglot': 10}, 'transit': {'CIFAR10': 10, 'Omniglot': 10},
                     'create': {'CIFAR10': 10, 'Omniglot': 10}}
    pivot = 'is'
    s = '#!/bin/bash\n'
    for filename in filenames:
        script_name = '{}.py'.format(filename)
        k = 0
        for data_name in summarized:
            summarized_d = summarized[data_name]
            for m in summarized_d:
                model_tag = summarized_d[m][pivot]['best_name']
                model_tag_list = model_tag.split('_')
                init_seed = model_tag_list[0]
                model_name = model_tag_list[3]
                if model_name == 'vqvae' or (filename == 'transit' and 'pixelcnn' in model_name):
                    continue
                if 'mc' in model_name:
                    control_name = '0.5'
                else:
                    control_name = 'None'
                controls = [init_seed, data_name, model_name, control_name, save_per_mode[filename][data_name]]
                s = s + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --init_seed {} --data_name {} --model_name {} ' \
                        '--control_name {} --save_per_mode {}&\n'.format(
                    gpu_ids[k % len(gpu_ids)], script_name, *controls)
                if k % round == round - 1:
                    s = s[:-2] + '\n'
                k = k + 1
    print(s)
    run_file = open('./make_img.sh', 'w')
    run_file.write(s)
    run_file.close()
    return


if __name__ == '__main__':
    main()