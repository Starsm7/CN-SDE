import time
import ot
import torch
from torch import optim, nn
import numpy as np
from geomloss import SamplesLoss
from src.emd import earth_mover_distance
from src.model import ForwardSDE, estimate_probs, derive_loss, KL_div, get_ROCKET
import os
from src.config_Veres import load_data


def p_samp(p, num_samp, w=None):
    repflag = p.shape[0] < num_samp
    p_sub = np.random.choice(p.shape[0], size=num_samp, replace=repflag)
    if w is None:
        w_ = torch.ones(len(p_sub))
    else:
        w_ = w[p_sub].clone()
    w_ = w_ / w_.sum()
    return p[p_sub, :].clone(), w_



def init_device(args):
    args.cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device('cuda:{}'.format(args.device) if args.cuda else 'cpu')
    return device


# ---- loss
class OTLoss():
    def __init__(self, config, device):
        self.ot_solver = SamplesLoss("sinkhorn", p=2, blur=config.sinkhorn_blur,
                                     scaling=config.sinkhorn_scaling, debias=True)
        self.device = device
    def __call__(self, a_i, x_i, b_j, y_j, requires_grad = True):
        a_i = a_i.to(self.device)
        x_i = x_i.to(self.device)
        b_j = b_j.to(self.device)
        y_j = y_j.to(self.device)
        if requires_grad:
            a_i.requires_grad_()
            x_i.requires_grad_()
            b_j.requires_grad_()
        loss_xy = self.ot_solver(a_i, x_i, b_j, y_j)
        return loss_xy

class OT_loss1(nn.Module):
    _valid = 'emd sinkhorn sinkhorn_knopp_unbalanced'.split()
    def __init__(self, which='emd', use_cuda=True):
        if which not in self._valid:
            raise ValueError(f'{which} not known ({self._valid})')
        self.which = which
        self.use_cuda = use_cuda
    def __call__(self, mu, source, nu, target, device='cuda', sigma=None, use_cuda=None):
        if use_cuda is None:
            use_cuda = self.use_cuda
        if not isinstance(mu, torch.Tensor):
            mu = torch.tensor(mu, dtype=torch.float32)
        if not isinstance(nu, torch.Tensor):
            nu = torch.tensor(nu, dtype=torch.float32)
        if use_cuda:
            mu = mu.to(device)
            nu = nu.to(device)
            target = target.to(device)
        M = torch.cdist(source, target) ** 2
        if self.which == 'emd':
            pi = ot.emd(mu.detach().cpu().numpy(), nu.detach().cpu().numpy(), M.detach().cpu().numpy())
        elif self.which == 'sinkhorn':
            if sigma is None:
                raise ValueError('sigma must be provided for sinkhorn method')
            pi = ot.sinkhorn(mu, nu, M, sigma)
        elif self.which == 'sinkhorn_knopp_unbalanced':
            if sigma is None:
                raise ValueError('sigma must be provided for sinkhorn_knopp_unbalanced method')
            pi = ot.unbalanced.sinkhorn_knopp_unbalanced(mu.detach().cpu().numpy(), nu.detach().cpu().numpy(),
                                                         M.detach().cpu().numpy(), sigma, sigma)
        else:
            raise ValueError(f'{self.which} not known ({self._valid})')

        if isinstance(pi, np.ndarray):
            pi = torch.tensor(pi, dtype=torch.float32)
        elif isinstance(pi, torch.Tensor):
            pi = pi.clone().detach()
        pi = pi.to(device) if use_cuda else pi
        M = M.to(pi.device)
        loss = torch.sum(pi * M)
        return loss

def estimate_probs_clc(clc_pred, temperature=0.1, sum_dim=1, normalize_dim=(1, 2)):
    probs = torch.softmax(clc_pred / temperature, dim=-1)
    probs = probs.sum(dim=sum_dim, keepdim=True)
    probs = probs / probs.sum(dim=normalize_dim, keepdim=True)
    return probs

def run(args, initial_config, leaveouts=None):
    # ---- initialize
    device = init_device(args)

    Train_ts = args.train_t
    args.leaveout_t = 'leaveout' + '&'.join(map(str, leaveouts))
    args.train_t = list(sorted(set(Train_ts) - set(leaveouts)))
    args.test_t = leaveouts
    print('--------------------------------------------')
    print('----------leaveout_t=', leaveouts, '---------')
    print('----------train_t=', args.train_t)
    print('--------------------------------------------')


    config = initial_config(args)
    x, y, c, config = load_data(config)

    roc_model = torch.load(f'data/models/Veres_ROCKET_{config.test_t[0]}_n_kernels=250.pt')
    Logistic = torch.jit.load(f"data/models/Veres_Logistic Regression{config.test_t[0]}_n_kernels=250.pt").float()
    c = torch.tensor(estimate_probs(c), requires_grad=True)
    LOSS0 = []
    for i in range(8 - 1):
        x_i = x[i+1]
        x_i_r = get_ROCKET(roc_model, x_i)
        clc_pred = Logistic.predict_proba(x_i_r)
        # class_indices = torch.argmax(clc_pred, dim=1)
        pred_probs = estimate_probs_clc(clc_pred, sum_dim=0, normalize_dim=(0, 1)).view(1, 12)
        loss0 = KL_div(pred_probs, c[i+1, :])
        LOSS0.append(loss0)

    if os.path.exists(os.path.join(config.out_dir, 'train.epoch_003000.pt')):
        print(os.path.join(config.out_dir, 'train.epoch_003000.pt'), ' exists. Skipping.')
        
    else:
        # cls
        model = ForwardSDE(config)
        print(model)

        # loss = OTLoss(config, device)
        loss = OT_loss1()
        torch.save(config.__dict__, config.config_pt)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.train_lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, min_lr=0.00001, patience=10, verbose=True)
        optimizer.zero_grad()

        # fit on time points
        best_train_loss_xy = np.inf

        end_tau, start_tau = 0.1, 1
        gumbel_tau_gamma = (end_tau / start_tau) ** (1 / config.train_epochs)
        gumbel_tau = start_tau

        with open(config.train_log, 'w') as log_handle:
            for epoch in range(config.train_epochs):
                a = time.time()
                losses_xy = []
                losses_w = []
                config.train_epoch = epoch
                dat_prev = x[config.start_t]
                x_i, a_i = p_samp(dat_prev, int(dat_prev.shape[0] * args.train_batch))
                x_i = x_i.to(device)
                ts = [0] + Train_ts
                y_ts = [np.float64(y[ts_i]) for ts_i in ts]
                x_s, sample_graph, loss_cuts = model(y_ts, x_i, gumbel_tau)  # 1 --> 8
#########################################################################################################################
                if config.clc == 1:
                    x_s_p = x_s.view(-1, x_s.shape[2])
                    x_clc = get_ROCKET(roc_model, x_s_p)
                    clc_pred = Logistic.predict_proba(x_clc)
                    pred_probs = estimate_probs_clc(clc_pred.reshape(x_s.shape[0], x_s.shape[1], 12)).view(x_s.shape[0], 12)
                loss_w = [x for x in derive_loss(ts, LOSS0, c, pred_probs)]
#########################################################################################################################
                if config.test_t[0] == 7:
                    weight = torch.tensor([1, 1, 1, 1, 1, 1.5, 1.5, 1]).cuda()
                elif config.test_t[0] == 6:
                    weight = torch.tensor([1, 1, 1, 1, 1, 1.5, 1, 1.5]).cuda()
                elif config.test_t[0] == 5:
                    weight = torch.tensor([1, 1, 1, 1, 1.5, 1, 1.5, 1]).cuda()
                elif config.test_t[0] == 4:
                    weight = torch.tensor([1, 1, 1, 1.5, 1, 1.5, 1, 1]).cuda()
                elif config.test_t[0] == 3:
                    weight = torch.tensor([1, 1, 1.5, 1, 1.5, 1, 1, 1]).cuda()
                elif config.test_t[0] == 2:
                    weight = torch.tensor([1, 1.5, 1, 1.5, 1, 1, 1, 1]).cuda()
                elif config.test_t[0] == 1:
                    weight = torch.tensor([1, 1, 1.5, 1.5, 1, 1, 1, 1]).cuda()

                for j in config.train_t:
                    t_cur = j
                    dat_cur = x[t_cur]
                    y_j, b_j = p_samp(dat_cur, int(dat_cur.shape[0] * args.train_batch))
                    position = Train_ts.index(j)
                    loss_xy = loss(a_i, x_s[position + 1], b_j, y_j) * weight[position + 1]
                    losses_xy.append(loss_xy.item())
                    losses_w.append(loss_w[position].item())
                    loss_all = loss_xy + loss_cuts + loss_w[position]
                    loss_all.backward(retain_graph=True)
                gumbel_tau *= gumbel_tau_gamma
                for j in config.test_t:
                    t_cur = j
                    dat_cur = x[t_cur]
                    y_j, b_j = p_samp(dat_cur, int(dat_cur.shape[0] * args.train_batch))
                    position = Train_ts.index(j)
                    loss_xy_test = loss(a_i, x_s[position+1], b_j, y_j)
                train_loss_xy = np.mean(losses_xy)
                train_loss_w = np.mean(losses_w)

                # step
                if config.train_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.train_clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                # report
                desc = "[train] {}".format(epoch + 1)
                desc += " loss_xy {:.6f}".format(train_loss_xy)
                if config.clc == 1:
                    desc += " loss_w {:.6f}".format(train_loss_w)
                    desc += " loss_cuts {:.6f}".format(loss_cuts.item())
                desc += " best_xy {:.6f}".format(best_train_loss_xy)
                desc += " test {:.6f}".format(loss_xy_test)

                if (config.train_epoch + 1) % config.save == 0:
                    epoch_ = str(config.train_epoch + 1).rjust(6, '0')
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'gumbel_tau': gumbel_tau,
                        'epoch': config.train_epoch + 1,
                    }, config.train_pt.format('epoch_{}'.format(epoch_)))

                if train_loss_xy < best_train_loss_xy:
                    best_train_loss_xy = train_loss_xy
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'gumbel_tau': gumbel_tau,
                        'epoch': config.train_epoch + 1,
                    }, config.train_pt.format(f'best'))
                    torch.save(sample_graph, os.path.join(config.out_dir, 'sample_graph.pt'))
                    desc += " save_best_xy"

                    if epoch >= 5000:
                        for j in config.test_t:
                            t_cur = j
                            evaluate_n = 2000
                            ns = 1000
                            x_0, _ = p_samp(x[0], evaluate_n)
                            x_0 = x_0.to(device)
                            x_s = []
                            for i in range(int(evaluate_n / ns)):
                                x_0_ = x_0[i * ns:(i + 1) * ns, ]
                                ts = [0, 1, 2, 3, 4, 5, 6, 7]
                                y_ts = [np.float64(y[ts_i]) for ts_i in ts]
                                _, x_s_, _, _, _ = model(y_ts, x_0_, gumbel_tau, roc_model, Logistic)
                                x_s.append(x_s_[t_cur].detach())
                            x_s = torch.cat(x_s)
                            y_t = x[t_cur]
                            loss_xy_emd = earth_mover_distance(x_s.cpu().numpy(), y_t)
                        desc += " test0 {:.6f}".format(loss_xy_emd)

                b = time.time()
                print(f"Time: {b-a:.2f} {desc}")
                log_handle.write(desc + '\n')
                log_handle.flush()

    return config

