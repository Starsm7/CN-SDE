import pandas as pd
import torch
from torch import nn
from tsai.data.core import TSDatasets, TSDataLoaders
from tsai.data.preprocessing import TSStandardize
from tsai.models.ROCKET_Pytorch import create_rocket_features
import src.sde as sde
import numpy as np
from torch import Tensor
import warnings
import torch.nn.functional as f
def get_ROCKET(roc_model, X_2d):
    y = torch.zeros(size=(X_2d.shape[0], 1))
    dsets = TSDatasets(X_2d, y, inplace=True)
    dls = TSDataLoaders.from_dsets(dsets, bs=768, drop_last=False, shuffle_train=False, device='cpu', batch_tfms=[TSStandardize(by_sample=True)], verbose=False)
    X_train, _ = create_rocket_features(dls.train, roc_model, verbose=False)
    return X_train
def gumbel_softmax(logits: Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1) -> Tensor:
    r"""
    Samples from the Gumbel-Softmax distribution (`Link 1`_  `Link alltime`_) and optionally discretizes.
    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.
    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.
    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.
    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`
      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)
    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> f.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> f.gumbel_softmax(logits, tau=1, hard=True)

    .. _Link 1:
        https://arxiv.org/abs/1611.00712
    .. _Link alltime:
        https://arxiv.org/abs/1611.01144
    """
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)
    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret
def estimate_probs(y, orders=['exo', 'fev_high_isl_low', 'neurog3_early', 'neurog3_late', 'neurog3_mid', 'phox2a', 'prog_nkx61', 'prog_sox2', 'sc_alpha', 'sc_beta', 'sc_ec', 'sst_hhex']):
    actual_probs = np.zeros((len(y), len(orders)))  # 初始化概率矩阵
    for i in range(len(y)):
        if isinstance(y, torch.Tensor):
            prob = pd.Series(y[i].numpy().flatten()).value_counts(normalize=True)
        else:
            prob = y[i].value_counts(normalize=True)
        order_prob = np.zeros(len(orders))
        k = 0
        for j in orders:
            if j in prob.index:
                order_prob[k] = prob[j]
            else:
                order_prob[k] = 0
            k += 1
        actual_probs[i, :] = order_prob
    return actual_probs
def derive_loss(Timepoints, loss0, Probs, sde_probs):
    Loss = []
    for i in range(len(Timepoints) - 1):
        actual_p2 = Probs[i + 1, :]
        loss = KL_div(sde_probs[i + 1], actual_p2) - loss0[i]
        # Loss.append(torch.log(f.relu(loss) * 100))
        Loss.append(f.relu(loss) * 10)
    return Loss
def KL_div(a, b, add_min=1e-10):
    a = a.clone()
    b = b.clone()
    a[a == 0] += add_min
    b[b == 0] += add_min
    div = torch.sum(a * torch.log(a / b))
    return div * 10
class MLP(nn.Module):
    def __init__(self, in_dim, n_hid, out_dim, n_layer):
        super(MLP, self).__init__()
        dims = (
            [(in_dim, n_hid)]
            + [(n_hid, n_hid) for _ in range(n_layer - 1)]
            + [(n_hid, out_dim)]
        )
        fc_layers = [nn.Linear(pair[0], pair[1]) for pair in dims]
        lr_layers = [nn.LeakyReLU(0.05) for _ in range(n_layer)]
        layers = []
        for i in range(n_layer):
            layers.append(fc_layers[i])
            layers.append(lr_layers[i])
        layers.append(fc_layers[-1])
        self.network = nn.Sequential(*layers)
        self.init_weights()
    def forward(self, x, t, mask):
        x_masked = torch.einsum("mn,nn->mn", x, mask)
        x_masked = torch.cat([x_masked, t], dim=1)
        y_pred = self.network(x_masked)
        return y_pred
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
# class ExpertGate(nn.Module):
#     def __init__(self, in_channels, n_expert, layer):
#         super(ExpertGate, self).__init__()
#         self.n_expert = n_expert
#         self.layer = layer
#         self.expert_layers = nn.ModuleList([MLP(in_channels, 400, in_channels-1, 3) for _ in range(n_expert)])
#     def forward(self, x, ts, mask):
#         expert_outputs = [expert(x, ts, mask) for expert in self.expert_layers]
#         e_net = []
#         for i in range(self.n_expert):
#             e_net.append(expert_outputs[i])
#         out = sum(e_net) / len(e_net)
#         return out
class ExpertGate(nn.Module):
    def __init__(self, in_channels, n_expert, layer):
        super(ExpertGate, self).__init__()
        self.n_expert = n_expert
        self.layer = layer
        self.expert_layers = nn.ModuleList([MLP(in_channels, 400, in_channels-1, 3) for _ in range(n_expert)])
        self.gate = f.softmax(nn.Parameter(torch.rand(self.n_expert)))
    def forward(self, x, ts, mask):
        expert_outputs = [expert(x, ts, mask) for expert in self.expert_layers]
        e_net = []
        for i in range(self.n_expert):
            e_net.append(expert_outputs[i] * self.gate[i])
        return sum(e_net)
class AutoGenerator(nn.Module):
    def __init__(self, config):
        super(AutoGenerator, self).__init__()
        self.dim = config.x_dim
        self.k_dims = config.k_dims
        self.layers = config.layers
        self.graph = nn.Parameter(torch.ones([self.dim, self.dim]))
        self.Expert_Gate = ExpertGate(in_channels=self.dim + 1, n_expert=3, layer=8)
        self.noise_type = 'diagonal'
        self.sde_type = "ito"
        self.sigma_const = config.sigma_const
        self.register_buffer('sigma', torch.as_tensor(self.sigma_const))
        self.sigma = self.sigma.repeat(self.dim).unsqueeze(0)
        self.losses = 0.0
    def set_model(self, gumbel_tau):
        self.gumbel_tau = gumbel_tau
    def set_cuts_losses(self, loss):
        self.losses = loss
    def get_cuts(self):
        return self.sample_graph, self.losses
    def graph_discov(self, x, ts, gumbel_tau):
        def sigmoid_gumbel_sample(graph, tau=1):
            prob = torch.sigmoid(graph.unsqueeze(2))
            logits = torch.concat([prob, (1 - prob)], axis=-1)
            samples = gumbel_softmax(logits, tau=tau)[:, :, 0]
            return samples
        loss = 0
        prob_graph = torch.sigmoid(self.graph)
        sample_graph = sigmoid_gumbel_sample(self.graph, tau=gumbel_tau) + self.graph
        graph_save = sigmoid_gumbel_sample(self.graph, tau=gumbel_tau)
        gs = prob_graph.shape
        loss = loss + torch.norm(prob_graph, p=1) / (gs[0] * gs[1])
        y_pred = self.Expert_Gate(x, ts, sample_graph)
        return y_pred, graph_save, loss
    def _pot(self, t, x):
        pot, self.sample_graph, loss = self.graph_discov(x, t, self.gumbel_tau)
        self.set_cuts_losses(loss)
        return pot
    def f(self, t, x):
        t = (((torch.ones(x.shape[0]).to(x.device)) * t).unsqueeze(1))
        pot = self._pot(t, x)
        return pot
    def g(self, t, x):
        g = self.sigma.repeat(x.shape[0], 1)
        return g

class ForwardSDE(torch.nn.Module):
    def __init__(self, config):
        super(ForwardSDE, self).__init__()
        self._func = AutoGenerator(config)
    def forward(self, ts, x_0, gumbel_tau):
        self._func.set_model(gumbel_tau)
        x_s_0 = sde.sdeint_adjoint(self._func, x_0, ts, method='euler', dt=0.1, dt_min=0.0001, adjoint_method='euler', names={'drift': 'f', 'diffusion': 'g'})
        sample_graph, loss = self._func.get_cuts()
        return x_s_0,  sample_graph, loss


