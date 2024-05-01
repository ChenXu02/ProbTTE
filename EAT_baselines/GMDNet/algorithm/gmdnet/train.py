from my_utils.utils import *
import numpy as np
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def seed_it(seed):
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)
seed_it(1024)


class GMDNet_Dataset(Dataset):
    def __init__(
            self,
            mode: str,
            params: dict,
    ) -> None:
        super().__init__()
        if mode not in ["train", "val", "test"]:
            raise ValueError
        path_key = {'train': 'train_path', 'val': 'val_path', 'test': 'test_path'}[mode]
        path = params[path_key]
        print(path)
        self.data = np.load(path, allow_pickle=True).item()

    def __len__(self):
        return len(self.data['label'])

    def __getitem__(self, index):
        route = self.data['route'][index]
        mask = self.data['mask'][index]
        f = self.data['f'][index]
        label = self.data['label'][index]

        return route, mask, f, label

def collate_fn(batch):
    return batch

def process_batch(batch, model, device, edge, node, A):
    route, mask, f, label = zip(*batch)
    route = torch.LongTensor(route).to(device)
    mask = torch.LongTensor(mask).to(device)
    f = torch.FloatTensor(f).to(device)
    label = torch.FloatTensor(label).to(device)
    edge = torch.FloatTensor(edge).to(device)
    node = torch.FloatTensor(node).to(device)
    A = torch.LongTensor(A).to(device)

    negative_log_likelihood = model.forward(route, mask, f, edge, node, A, label, 'train')

    return negative_log_likelihood

def test_model(model, test_dataloader, device, edge, node, A):
    predicts_list = []
    label_list = []
    log_likelihood_list = []
    crps_list = []
    model.eval()
    edge = torch.FloatTensor(edge).to(device)
    node = torch.FloatTensor(node).to(device)
    A = torch.LongTensor(A).to(device)
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            route, mask, f, label = zip(*batch)
            route = torch.LongTensor(route).to(device)
            mask = torch.LongTensor(mask).to(device)
            f = torch.FloatTensor(f).to(device)
            label = torch.FloatTensor(label).to(device)

            pi, mu, sigma = model.forward(route, mask, f, edge, node, A, label, 'test')
            m = torch.distributions.Normal(loc=mu, scale=sigma)
            pred = torch.sum(m.mean * pi, dim=1).unsqueeze(1)

            log_likelihood = torch.mean(torch.mul(m.log_prob(label.float()), pi))
            log_likelihood_list += [log_likelihood.cpu().item()]
            predicts_list += pred.detach().cpu().tolist()
            label_list += label.detach().cpu().tolist()
            crps_metric = CRPSMetric(label, mu, sigma)
            get_crps = crps_metric.gaussian_crps()
            crps = torch.sum(torch.mul(get_crps, pi), dim=1)
            crps_list.append(crps.detach().cpu().numpy().mean())

        log_likelihood = np.array(log_likelihood_list)
        predicts = np.array(predicts_list)
        labels = np.array(label_list)
        CRPS = np.array(crps_list)

        from sklearn.metrics import mean_absolute_error as mae
        from sklearn.metrics import mean_absolute_percentage_error as mape

        print('mape:', mape(labels, predicts))
        print('mae:', mae(labels, predicts))
        print('negative-log-likelihood:', log_likelihood.mean())
        print('CRPS:', CRPS.mean())

        val_mae = mae(labels, predicts)
        val_mape = mape(labels, predicts)
        return val_mae, val_mape, log_likelihood.mean(), CRPS.mean()

def get_params():
    from my_utils.utils import get_common_params
    parser = get_common_params()
    # Model parameters
    parser.add_argument('--model', type=str, default='GMDNet')
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--n_gaussians', type=int, default=5)
    parser.add_argument('--is_eval', type=str, default=True, help='True means load existing model')
    parser.add_argument('--att_hidden_size', type=int, default=64, help='dims of route sa hidden dim')
    parser.add_argument('--num_of_attention_heads', type=int, default=8, help='dims of route sa hidden dim')
    parser.add_argument('--num_layers', type=int, default=2, help='num of gnn layer')
    parser.add_argument('--dirichlet_alpha', type = int, default=1, help='Dirichlet regularizer')

    args, _ = parser.parse_known_args()

    return args

from my_utils.utils import run

def main(params):
    run(params, GMDNet_Dataset, process_batch, test_model, collate_fn)

if __name__ == '__main__':
    import time, nni, torch
    import logging

    print('lets start')
    logger = logging.getLogger('training')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    try:
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(get_params())
        params.update(tuner_params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
