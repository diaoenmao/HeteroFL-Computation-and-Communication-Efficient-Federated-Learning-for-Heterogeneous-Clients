import torch
import torch.nn.functional as F

from utils import recur


def Accuracy(output, target, topk=1):
    with torch.no_grad():
        batch_size = target.size(0)
        pred_k = output.topk(topk, 1, True, True)[1]
        correct_k = pred_k.eq(target.view(-1, 1).expand_as(pred_k)).float().sum()
        acc = (correct_k * (100.0 / batch_size)).item()
    return acc


def Perplexity(output, target):
    with torch.no_grad():
        ce = F.cross_entropy(output, target)
        perplexity = torch.exp(ce).item()
    return perplexity


class Metric(object):
    def __init__(self):
        self.metric = {'Loss': (lambda input, output: output['loss'].item()),
                       'Local-Loss': (lambda input, output: output['loss'].item()),
                       'Global-Loss': (lambda input, output: output['loss'].item()),
                       'Accuracy': (lambda input, output: recur(Accuracy, output['score'], input['label'])),
                       'Local-Accuracy': (lambda input, output: recur(Accuracy, output['score'], input['label'])),
                       'Global-Accuracy': (lambda input, output: recur(Accuracy, output['score'], input['label'])),
                       'Perplexity': (lambda input, output: recur(Perplexity, output['score'], input['nsymbol']))}

    def evaluate(self, metric_names, input, output):
        evaluation = {}
        for metric_name in metric_names:
            evaluation[metric_name] = self.metric[metric_name](input, output)
        return evaluation