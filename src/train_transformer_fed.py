import argparse
import copy
import datetime
import models
import numpy as np
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
from config import cfg
from data import fetch_dataset, split_dataset, SplitDataset, BatchDataset
from fed import Federation
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume
from logger import Logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
for k in cfg:
    cfg[k] = args[k]
if args['control_name']:
    cfg['control'] = {k: v for k, v in zip(cfg['control'].keys(), args['control_name'].split('_'))} \
        if args['control_name'] != 'None' else {}
cfg['control_name'] = '_'.join([cfg['control'][k] for k in cfg['control']])
cfg['pivot_metric'] = 'Global-Perplexity'
cfg['pivot'] = float('inf')
cfg['metric_name'] = {'train': {'Local': ['Local-Loss', 'Local-Perplexity']},
                      'test': {'Global': ['Global-Loss', 'Global-Perplexity']}}


def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['subset'], cfg['model_name'], cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    seed = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dataset = fetch_dataset(cfg['data_name'], cfg['subset'])
    process_dataset(dataset)
    model = eval('models.{}(model_rate=cfg["global_model_rate"]).to(cfg["device"])'.format(cfg['model_name']))
    optimizer = make_optimizer(model, cfg['lr'])
    scheduler = make_scheduler(optimizer)
    if cfg['resume_mode'] == 1:
        last_epoch, data_split, label_split, model, optimizer, scheduler, logger = resume(model, cfg['model_tag'],
                                                                                          optimizer, scheduler)
    elif cfg['resume_mode'] == 2:
        last_epoch = 1
        _, data_split, label_split, model, _, _, _ = resume(model, cfg['model_tag'])
        logger_path = os.path.join('output', 'runs', '{}'.format(cfg['model_tag']))
        logger = Logger(logger_path)
    else:
        last_epoch = 1
        data_split, label_split = split_dataset(dataset, cfg['num_users'], cfg['data_split_mode'])
        logger_path = os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag']))
        logger = Logger(logger_path)
    if data_split is None:
        data_split, label_split = split_dataset(dataset, cfg['num_users'], cfg['data_split_mode'])
    global_parameters = model.state_dict()
    federation = Federation(global_parameters, cfg['model_rate'], label_split)
    for epoch in range(last_epoch, cfg['num_epochs']['global'] + 1):
        logger.safe(True)
        train(dataset['train'], data_split['train'], label_split, federation, model, optimizer, logger, epoch)
        test_model = model
        test(dataset['test'], test_model, logger, epoch)
        if cfg['scheduler_name'] == 'ReduceLROnPlateau':
            scheduler.step(metrics=logger.mean['train/{}'.format(cfg['pivot_metric'])])
        else:
            scheduler.step()
        logger.safe(False)
        model_state_dict = model.state_dict()
        save_result = {
            'cfg': cfg, 'epoch': epoch + 1, 'data_split': data_split, 'label_split': label_split,
            'model_dict': model_state_dict, 'optimizer_dict': optimizer.state_dict(),
            'scheduler_dict': scheduler.state_dict(), 'logger': logger}
        save(save_result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if cfg['pivot'] > logger.mean['test/{}'.format(cfg['pivot_metric'])]:
            cfg['pivot'] = logger.mean['test/{}'.format(cfg['pivot_metric'])]
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    logger.safe(False)
    return


def train(dataset, data_split, label_split, federation, global_model, optimizer, logger, epoch):
    global_model.load_state_dict(federation.global_parameters)
    global_model.train(True)
    local, local_parameters, user_idx, param_idx = make_local(dataset, data_split, label_split, federation)
    num_active_users = len(local)
    lr = optimizer.param_groups[0]['lr']
    start_time = time.time()
    for m in range(num_active_users):
        local_parameters[m] = copy.deepcopy(local[m].train(local_parameters[m], lr, logger))
        if m % int((num_active_users * cfg['log_interval']) + 1) == 0:
            local_time = (time.time() - start_time) / (m + 1)
            epoch_finished_time = datetime.timedelta(seconds=local_time * (num_active_users - m - 1))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['num_epochs']['global'] - epoch) * local_time * num_active_users))
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * m / num_active_users),
                             'ID: {}({}/{})'.format(user_idx[m], m + 1, num_active_users),
                             'Learning rate: {}'.format(lr),
                             'Rate: {}'.format(federation.model_rate[user_idx[m]]),
                             'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            logger.write('train', cfg['metric_name']['train']['Local'])
    federation.combine(local_parameters, param_idx, user_idx)
    global_model.load_state_dict(federation.global_parameters)
    return


def test(dataset, model, logger, epoch):
    with torch.no_grad():
        metric = Metric()
        model.train(False)
        batch_dataset = BatchDataset(dataset, cfg['bptt'])
        for i, input in enumerate(batch_dataset):
            input_size = input['label'].size(0)
            input = to_device(input, cfg['device'])
            output = model(input)
            output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
            evaluation = metric.evaluate(cfg['metric_name']['test']['Global'], input, output)
            logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {}'.format(cfg['model_tag']),
                         'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        logger.write('test', cfg['metric_name']['test']['Global'])
    return


def make_local(dataset, data_split, label_split, federation):
    num_active_users = int(np.ceil(cfg['frac'] * cfg['num_users']))
    user_idx = torch.arange(cfg['num_users'])[torch.randperm(cfg['num_users'])[:num_active_users]].tolist()
    local_parameters, param_idx = federation.distribute(user_idx)
    local = [None for _ in range(num_active_users)]
    for m in range(num_active_users):
        model_rate_m = federation.model_rate[user_idx[m]]
        dataset_m = SplitDataset(dataset, data_split[user_idx[m]])
        local[m] = Local(model_rate_m, dataset_m, label_split[user_idx[m]])
    return local, local_parameters, user_idx, param_idx


class Local:
    def __init__(self, model_rate, dataset, label_split):
        self.model_rate = model_rate
        self.dataset = BatchDataset(dataset, cfg['bptt'])
        self.label_split = label_split

    def train(self, local_parameters, lr, logger):
        metric = Metric()
        model = eval('models.{}(model_rate=self.model_rate).to(cfg["device"])'.format(cfg['model_name']))
        model.load_state_dict(local_parameters)
        model.train(True)
        optimizer = make_optimizer(model, lr)
        for local_epoch in range(1, cfg['num_epochs']['local'] + 1):
            for i, input in enumerate(self.dataset):
                input_size = input['label'].size(0)
                input['label_split'] = torch.tensor(self.label_split)
                input = to_device(input, cfg['device'])
                optimizer.zero_grad()
                output = model(input)
                output['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                evaluation = metric.evaluate(cfg['metric_name']['train']['Local'], input, output)
                logger.append(evaluation, 'train', n=input_size)
        local_parameters = model.state_dict()
        return local_parameters


if __name__ == "__main__":
    main()