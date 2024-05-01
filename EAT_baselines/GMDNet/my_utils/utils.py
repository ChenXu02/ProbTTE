import argparse
import torch
from torch.optim import Adam
from pprint import pprint
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
"""
GMDNet is implemented in Python using Pytorch on Hygon C86 7151 CPUs and NVIDIA RTX A4000 GPUs, and the operating system is Ubuntu 20.04.1.
"""

def get_workspace():
    cur_path = os.path.abspath(__file__)
    file = os.path.dirname(cur_path)
    file = os.path.dirname(file)
    return file

ws = get_workspace()

class CRPSMetric:
    def __init__(self, x, loc, scale):
        self.value = x
        self.loc = loc
        self.scale = scale
    def gaussian_pdf(self, x):
        """
        Probability density function of a univariate standard Gaussian distribution with zero mean and unit variance.
        """
        _normconst = 1.0 / math.sqrt(2.0 * math.pi)
        return _normconst * torch.exp(-(x * x) / 2.0)
    def gaussian_cdf(self, x):
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))
    def gaussian_crps(self):
        sx = (self.value - self.loc) / self.scale
        pdf = self.gaussian_pdf(sx)
        cdf = self.gaussian_cdf(sx)
        pi_inv = 1.0 / math.sqrt(math.pi)
        # the actual crps
        crps = self.scale * (sx * (2 * cdf - 1) + 2 * pdf - pi_inv)
        return crps

def dir_check(path):
    """
    check weather the dir of the given path exists, if not, then create it
    """
    import os
    dir = path if os.path.isdir(path) else os.path.split(path)[0]
    if not os.path.exists(dir): os.makedirs(dir)


def save2file_meta(params, file_name, head):
    def timestamp2str(stamp):
        utc_t = int(stamp)
        utc_h = utc_t // 3600
        utc_m = (utc_t // 60) - utc_h * 60
        utc_s = utc_t % 60
        hour = (utc_h + 8) % 24
        t = f'{hour}:{utc_m}:{utc_s}'
        return t

    import csv, time, os
    dir_check(file_name)
    if not os.path.exists(file_name):
        f = open(file_name, "w", newline='\n')
        csv_file = csv.writer(f)
        csv_file.writerow(head)
        f.close()
    # write_to_hdfs(file_name, head)
    with open(file_name, "a", newline='\r') as file:  # linux:\n    windows:\r\n    mac:\r
        csv_file = csv.writer(file)
        params['time'] = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        params['log_time'] = timestamp2str(time.time())
        data = [params[k] for k in head]
        csv_file.writerow(data)


def whether_stop(metric_lst=[], n=2, mode='minimize'):

    if len(metric_lst) < 1: return False  # at least have 2 results.
    if mode == 'minimize': metric_lst = [-x for x in metric_lst]
    max_v = max(metric_lst)
    max_idx = 0
    for idx, v in enumerate(metric_lst):
        if v == max_v: max_idx = idx
    return max_idx < len(metric_lst) - n


class EarlyStop():
    def __init__(self, mode='maximize', patience=1):
        self.mode = mode
        self.patience = patience
        self.metric_lst = []
        self.stop_flag = False
        self.best_epoch = -1  # the best epoch
        self.is_best_change = False  # whether the best change compare to the last epoch

    def append(self, x):
        self.metric_lst.append(x)
        # update the stop flag
        self.stop_flag = whether_stop(self.metric_lst, self.patience, self.mode)
        # update the best epoch
        best_epoch = self.metric_lst.index(max(self.metric_lst)) if self.mode == 'maximize' else self.metric_lst.index(
            min(self.metric_lst))
        if best_epoch != self.best_epoch:
            self.is_best_change = True
            self.best_epoch = best_epoch  # update the wether best change flag
        else:
            self.is_best_change = False
        return self.is_best_change

    def best_metric(self):
        if len(self.metric_lst) == 0:
            return -1
        else:
            return self.metric_lst[self.best_epoch]


def get_common_params():
    # Training settings
    parser = argparse.ArgumentParser(description='Entry Point of the code')
    parser.add_argument('--dataset', type=str, default='logistics')
    ## common settings for deep models
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default:32)')
    parser.add_argument('--num_epoch', type=int, default=200, help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 1e-3)')
    parser.add_argument('--wd', type=float, default=1e-5, help='weight decay (default: 1e-5)')
    parser.add_argument('--early_stop', type=int, default=10, help='early stop at')
    parser.add_argument('--is_test', type=bool, default=False)

    return parser


def get_dataset_path(params):

    train_path = ws + '/data/train.npy'
    val_path =  ws + '/data/val.npy'
    test_path =  ws + '/data/test.npy'

    return train_path, val_path, test_path


def get_model_function(model):
    import algorithm.gmdnet.network as gmdnet

    model_dict = {
        'GMDNet': gmdnet.GMDNet
    }

    model = model_dict[model]
    return model


# import nni
def save2file(params):
    file_name = ws + f'/output/{params["model"]}.csv'
    head = [
        # data setting
        'dataset',
        # mdoel parameters
        'model', 'early_stop', 'hidden_dim', 'num_layers', 'att_hidden_size', 'n_gaussians', 'num_of_attention_heads', 'dirichlet_alpha',
        # training set
        'batch_size', 'time',
        # metric result
        'mae', 'mape', 'log-likelihood', 'crps'
    ]
    save2file_meta(params, file_name, head)


def train_val_test(train_loader, val_loader, test_loader, model, device, PROCESS_BATCH, TEST_MODEL, params,
                   train_edge, train_node, train_A, val_edge, val_node, val_A, test_edge, test_node, test_A):

    model.to(device)
    optimizer = Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])
    early_stop = EarlyStop(mode='minimize', patience=params['early_stop'])

    model_name = model.model_file_name() + f'{time.time()}'
    model_path = ws + f'/params/{model_name}'
    dir_check(model_path)

    for epoch in range(params['num_epoch']):
        if early_stop.stop_flag: break
        postfix = {"epoch": epoch, "loss": 0.0, "current_loss": 0.0}
        with tqdm(train_loader, total=len(train_loader), postfix=postfix) as t:
            print('train_data_loaded')
            ave_loss = None
            model.train()
            for i, batch in enumerate(t):
                loss = PROCESS_BATCH(batch, model, device, train_edge, train_node, train_A)

                if ave_loss is None:
                    ave_loss = loss.item()
                else:
                    ave_loss = ave_loss * i / (i + 1) + loss.item() / (i + 1)
                postfix["loss"] = ave_loss
                postfix["current_loss"] = loss.item()
                t.set_postfix(**postfix)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        val_mae, val_mape, val_log, val_crps = TEST_MODEL(model, val_loader, device, val_edge, val_node, val_A)
        is_best_change = early_stop.append(val_mae)

        if is_best_change:
            print('current mae:', val_mae, 'current mae:', early_stop.best_metric(), 'current log:', val_log, 'current crps:', val_crps)
            torch.save(model.state_dict(), model_path)
            print('best model saved')
            print('model path:', model_path)
            if params['is_test']: break
    try:
        print('loaded model path:', model_path)
        model.load_state_dict(torch.load(model_path))
        print('best model loaded !!!')
    except:
        print('load best model failed')
    test_mae, test_mape, test_log, test_crps = TEST_MODEL(model, test_loader, device, test_edge, test_node, test_A)
    params['model'] = model_name
    params['mae'] = test_mae
    params['mape'] = test_mape
    params['log-likelihood'] = test_log
    params['crps'] = test_crps
    save2file(params)

    print('\n-------------------------------------------------------------')
    print('Best epoch: ', early_stop.best_epoch)
    print(' Evaluation in test:', test_mae)

    return params


def run(params, DATASET, PROCESS_BATCH, TEST_MODEL, collate_fn=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params['device'] = device
    print(device)

    params['train_path'], params['val_path'], params['test_path'] = get_dataset_path(params)
    pprint(params)  # print the parameters

    train_dataset = DATASET(mode='train', params=params)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, collate_fn=collate_fn)
    train_edge = train_dataset.data['edge']
    train_node = train_dataset.data['node']
    train_A = train_dataset.data['A']

    val_dataset = DATASET(mode='val', params=params)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=True,
                            collate_fn=collate_fn)
    val_edge = val_dataset.data['edge']
    val_node = val_dataset.data['node']
    val_A = val_dataset.data['A']

    test_dataset = DATASET(mode='test', params=params)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=True,
                             collate_fn=collate_fn)
    test_edge = test_dataset.data['edge']
    test_node = test_dataset.data['node']
    test_A = test_dataset.data['A']

    # train, valid and test
    net_models = ['GMDNet']
    model = get_model_function(params['model'])
    model = model(params)
    print(model)
    if params['model'] in net_models:
        result_dict = train_val_test(train_loader, val_loader, test_loader, model, device, PROCESS_BATCH, TEST_MODEL,
                                     params, train_edge, train_node, train_A, val_edge, val_node, val_A, test_edge,
                                     test_node, test_A)
    else:
        pass

    return params


if __name__ == '__main__':
    pass