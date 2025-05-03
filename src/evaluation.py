from src.config_Veres import load_data
from src.model import ForwardSDE
import torch
import numpy as np
import pandas as pd
from geomloss import SamplesLoss 
import glob
import os
import src.fit_epoch as train
from src.emd import earth_mover_distance
from types import SimpleNamespace

def init_device(args):
    args.cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device('cuda:{}'.format(args.device) if args.cuda else 'cpu')

    return device

def derive_model(args, ckpt_name='epoch_003000'):
    device = init_device(args)


    x, y, _, config = load_data(args)
    model = ForwardSDE(config)

    train_pt = "./" + config.train_pt.format(ckpt_name)
    checkpoint = torch.load(train_pt)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    return model, x, y, device


def evaluate_fit(args, initial_config, use_loss='emd'):
    
    device = init_device(args)


    args.out_dir = 'RESULTS/' + args.data

    config = initial_config(args)
    x, y, c, config = load_data(config)
    config = SimpleNamespace(**torch.load(config.config_pt))

    file_info = 'interpolate-' + use_loss + '.log'
    log_path = os.path.join(config.out_dir, file_info)
    
    # if os.path.exists(log_path):
    #     print(log_path, 'exists. Skipping.')
    #     return

    losses_xy = []
    train_pts = sorted(glob.glob(config.train_pt.format('*')))
    print(config.train_pt)
    print(train_pts)
    for train_pt in train_pts:
        model = ForwardSDE(config)
        checkpoint = torch.load(train_pt)
        print('Loading model from {}'.format(train_pt))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        gumbel_tau = checkpoint['gumbel_tau']
        epoch = checkpoint['epoch']
        print(epoch)
        name = os.path.basename(train_pt).split('.')[1]
        roc_model = torch.load(f'data/models/Veres_ROCKET_{config.test_t[0]}_n_kernels=250.pt')
        Logistic = torch.jit.load(f"data/models/Veres_Logistic Regression{config.test_t[0]}_n_kernels=250.pt").float()

        # for t in config.train_t:
        #     loss_xy = _evaluate_impute_model(config, t, model, x, y, c, gumbel_tau, device, use_loss).item()
        #     losses_xy.append((name, 'train', y[t], loss_xy))
        try:
            for t in config.test_t: 
                loss_xy = _evaluate_impute_model(config, t, model, x, y, c, gumbel_tau, roc_model, Logistic, device, use_loss).item()
                losses_xy.append((name, 'test', y[t], loss_xy))
        except AttributeError:
            continue

    losses_xy = pd.DataFrame(losses_xy, columns = ['epoch', 'eval', 't', 'loss'])
    losses_xy.to_csv(log_path, sep = '\t', index = False)
    print(losses_xy)
    print('Wrote results to', log_path)
    

def _evaluate_impute_model(config, t_cur, model, x, y, c, gumbel_tau, roc_model, Logistic, device, use_loss='emd'):
    torch.manual_seed(0)
    np.random.seed(0)

    ot_solver = SamplesLoss("sinkhorn", p=2, blur=config.sinkhorn_blur,
                            scaling=config.sinkhorn_scaling)

    x_0, a_j = train.p_samp(x[0], config.evaluate_n)
    x_0 = x_0.to(device)

    x_s = []
    ####################################################################################################################

    for i in range(int(config.evaluate_n / config.ns)):
        x_0_ = x_0[i * config.ns:(i + 1) * config.ns, ]
        ts = [0, 1, 2, 3, 4, 5, 6, 7]
        y_ts = [np.float64(y[ts_i]) for ts_i in ts]
        x_s_, _, _ = model(y_ts, x_0_, gumbel_tau)
        x_s.append(x_s_[t_cur].detach())

    x_s = torch.cat(x_s)
    y_t = x[t_cur]
    print('y_t', y_t.shape)

    if use_loss == 'ot':
        loss_xy = ot_solver(x_s.contiguous(), y_t.contiguous().to(device))
    elif use_loss == 'emd':
        loss_xy = earth_mover_distance(x_s.cpu().numpy(), y_t)

    return loss_xy















































        
