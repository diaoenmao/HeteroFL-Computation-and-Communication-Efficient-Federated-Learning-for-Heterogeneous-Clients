import argparse
import datetime
import os
import torch
import torch.backends.cudnn as cudnn
import models
from config import cfg
from data import fetch_dataset, make_data_loader, SplitDataset
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, resume, collate
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
cfg['metric_name'] = {'train': {'Local': ['Local-Loss', 'Local-Accuracy']},
                      'test': {'Local': ['Local-Loss', 'Local-Accuracy'], 'Global': ['Global-Loss', 'Global-Accuracy']}}


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
    cfg['batch_size']['train'] = cfg['batch_size']['test']
    seed = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dataset = fetch_dataset(cfg['data_name'], cfg['subset'])
    process_dataset(dataset)
    model = eval('models.{}(model_rate=cfg["global_model_rate"], track=True).to(cfg["device"]).to(cfg["device"])'
                 .format(cfg['model_name']))
    last_epoch, data_split, label_split, model, _, _, _ = resume(model, cfg['model_tag'], load_tag='best', strict=False)
    logger_path = 'output/runs/test_{}'.format(cfg['model_tag'])
    test_logger = Logger(logger_path)
    test_logger.safe(True)
    stats(dataset['train'], model)
    test(dataset['test'], data_split['test'], label_split, model, test_logger, last_epoch)
    test_logger.safe(False)
    _, _, _, _, _, _, train_logger = resume(model, cfg['model_tag'], load_tag='checkpoint', strict=False)
    save_result = {'cfg': cfg, 'epoch': last_epoch, 'logger': {'train': train_logger, 'test': test_logger}}
    save(save_result, './output/result/{}.pt'.format(cfg['model_tag']))
    return


def stats(dataset, model):
    with torch.no_grad():
        data_loader = make_data_loader({'train': dataset})['train']
        model.train(True)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input = to_device(input, cfg['device'])
            model(input)
    return


def test(dataset, data_split, label_split, model, logger, epoch):
    with torch.no_grad():
        metric = Metric()
        model.train(False)
        for m in range(cfg['num_users']):
            data_loader = make_data_loader({'test': SplitDataset(dataset, data_split[m])})['test']
            for i, input in enumerate(data_loader):
                input = collate(input)
                input_size = input['img'].size(0)
                input['label_split'] = torch.tensor(label_split[m])
                input = to_device(input, cfg['device'])
                output = model(input)
                output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
                evaluation = metric.evaluate(cfg['metric_name']['test']['Local'], input, output)
                logger.append(evaluation, 'test', input_size)
        data_loader = make_data_loader({'test': dataset})['test']
        for i, input in enumerate(data_loader):
            input = collate(input)
            input_size = input['img'].size(0)
            input = to_device(input, cfg['device'])
            output = model(input)
            output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
            evaluation = metric.evaluate(cfg['metric_name']['test']['Global'], input, output)
            logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        logger.write('test', cfg['metric_name']['test']['Local'] + cfg['metric_name']['test']['Global'])
    return


if __name__ == "__main__":
    main()