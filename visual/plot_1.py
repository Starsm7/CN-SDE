import pandas as pd
import random
import seaborn as sns
sns.color_palette("bright")
import os
from types import SimpleNamespace
import numpy as np
import torch
from joblib import load
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from src.config_Veres import load_data
from src.model import ForwardSDE
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
def p_samp(p, num_samp, w=None):
    repflag = p.shape[0] < num_samp
    p_sub = np.random.choice(p.shape[0], size=num_samp, replace=repflag)
    if w is None:
        w_ = torch.ones(len(p_sub))
    else:
        w_ = w[p_sub].clone()
    w_ = w_ / w_.sum()
    return p[p_sub, :].clone(), w_
seed_everything(37)
path = "RESULTS/Veres/alltime/config.pt"
config = SimpleNamespace(**torch.load(path))
x, y, c, config = load_data(config)
sigma = 0.25
model = ForwardSDE(config).to(config.device)
checkpoint = torch.load('RESULTS/Veres/alltime/train.best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
gumbel_tau = checkpoint['gumbel_tau']
model.eval()
#######################################################################################################################

ts = [0] + config.train_t
y_ts = [np.float64(y[ts_i]) for ts_i in ts]
x_i = x[0]
r_i = torch.zeros(x_i.shape[0]).unsqueeze(1)
x_r_i = torch.cat([x_i, r_i], dim=1)
x_r_i = x_r_i.to(config.device)
x_r_s, sample_graph, loss_cuts = model(y_ts, x_r_i, gumbel_tau)
x_r_s[0, :, 0:-1] = x_i
sample_time = np.linspace(np.min(y_ts), np.max(y_ts), num=3)
sample_time = [np.float64(ts_i) for ts_i in sample_time]
x_i_p, _ = p_samp(x[0], 50)
r_i_p = torch.zeros(x_i_p.shape[0]).unsqueeze(1)
x_r_i_p = torch.cat([x_i_p, r_i_p], dim=1)
x_r_i_p = x_r_i_p.to(config.device)
trajectories, _, _ = model(sample_time, x_r_i_p, gumbel_tau)
trajectories[0, :, 0:-1] = x_i_p
trajectories = trajectories[:, :, 0:-1]

if not os.path.exists('x_preds.npz') or not os.path.exists('x_trues.npz') or not os.path.exists('trajectories.npy'):
    um = load(r'data/data/Veres/alltime/um_transformer.joblib')
    trajectories_2dim = np.array([um.transform(trajectories[i, :, 0:-1].cpu().detach().numpy()) for i in
                         range(trajectories.size(0))])
    preds = [um.transform(x_r_s[i, :, 0:-1].cpu().detach().numpy()) for i in range(x_r_s.size(0))]
    print(preds[0].shape)
    trues = [um.transform(x_.cpu().detach().numpy()) for x_ in x]
    np.save('trajectories.npy', trajectories_2dim)
    np.savez('x_preds.npz', *preds)
    np.savez('x_trues.npz', *trues)
else:
    trajectories_2dim = np.load('trajectories.npy')
    loaded_data = np.load('x_preds.npz')
    preds = [loaded_data[f] for f in loaded_data.files]
    loaded_data = np.load('x_trues.npz')
    trues = [loaded_data[f] for f in loaded_data.files]
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange']
labels = [f"Time {i + 1}" for i in range(3)]
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
for i, (true, color, label) in enumerate(zip(trues, colors, labels)):
    axes[0].scatter(true[:, 0], true[:, 1], color=color, marker='o', alpha=0.5,
                    label=f'True {label}')
axes[0].set_title("Truth")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].legend()
axes[0].grid(True)
for i, (pred, color, label) in enumerate(zip(preds, colors, labels)):
    axes[1].scatter(pred[:, 0], pred[:, 1], color=color, marker='x', alpha=0.7,
                    label=f'Pred {label}')
axes[1].set_title("Prediction")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].legend()
axes[1].grid(True)
all_x = [point[0] for sublist in trues + preds for point in sublist]
all_y = [point[1] for sublist in trues + preds for point in sublist]
x_min, x_max = min(all_x), max(all_x)
y_min, y_max = min(all_y), max(all_y)
axes[0].set_xlim(x_min, x_max)
axes[0].set_ylim(y_min, y_max)
axes[1].set_xlim(x_min, x_max)
axes[1].set_ylim(y_min, y_max)
axes[0].set_aspect('equal')
axes[1].set_aspect('equal')
plt.tight_layout()
plt.show()
#####################################################################################################################
if not os.path.exists('true.csv'):
    df = pd.DataFrame(columns=['samples', 'x1', 'x2'])
    for label, true in enumerate(trues):
        for i in range(true.shape[0]):
            df = df.append({'samples': label, 'x1': true[i][0], 'x2': true[i][1]}, ignore_index=True)
    df.to_csv('true.csv', index=False)
else:
    df = pd.read_csv('true.csv')
print(df)
def plot_combined_data(df, generated):
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, generated.shape[0]))
    plt.figure(figsize=(12, 8))
    for i, label in enumerate(df['samples'].unique()):
        subset = df[df['samples'] == label]
        x = subset['x1']
        y = subset['x2']
        plt.scatter(x, y, label=f'Time {label} (df)', alpha=0.1, color=colors[i], marker='X')
    for i in range(generated.shape[0]):
        x = generated[i, :, 0]
        y = generated[i, :, 1]
        plt.scatter(x, y, color=colors[i], label=f'Time {i} (generated)', alpha=0.7)
    plt.xlim(min(df['x1'].min(), generated[:, :, 0].min()), max(df['x1'].max(), generated[:, :, 0].max()))
    plt.ylim(min(df['x2'].min(), generated[:, :, 1].min()), max(df['x2'].max(), generated[:, :, 1].max()))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Combined Scatter Plot of df and Generated Data')
    plt.show()
plot_combined_data(df, np.stack(preds,axis=0))
#####################################################################################################################
def to_np(data):
    return data.detach().cpu().numpy()
def new_plot_comparisions(
        df, generated, trajectories,
        palette='viridis',
        df_time_key='samples',
        x='d1', y='d2', z='d3',
        groups=None,
        save=False, path='.', file='comparision.png',
        is_3d=False
):
    if groups is None:
        groups = sorted(df[df_time_key].unique())
    cmap = plt.cm.viridis
    sns.set_palette(palette)
    plt.rcParams.update({
        'axes.prop_cycle': plt.cycler(color=cmap(np.linspace(0, 1, len(groups) + 1))),
        'axes.axisbelow': False,
        'axes.edgecolor': 'lightgrey',
        'axes.facecolor': 'None',
        'axes.grid': False,
        'axes.labelcolor': 'dimgrey',
        'axes.spines.right': False,
        'axes.spines.top': False,
        'figure.facecolor': 'white',
        'lines.solid_capstyle': 'round',
        'patch.edgecolor': 'w',
        'patch.force_edgecolor': True,
        'text.color': 'dimgrey',
        'xtick.bottom': False,
        'xtick.color': 'dimgrey',
        'xtick.direction': 'out',
        'xtick.top': False,
        'ytick.color': 'dimgrey',
        'ytick.direction': 'out',
        'ytick.left': False,
        'ytick.right': False,
        'font.size': 12,
        'axes.titlesize': 10,
        'axes.labelsize': 12
    })

    n_cols = 1
    n_rols = 1

    grid_figsize = [12, 8]
    dpi = 300
    grid_figsize = (grid_figsize[0] * n_cols, grid_figsize[1] * n_rols)
    fig = plt.figure(None, grid_figsize, dpi=dpi)

    hspace = 0.3
    wspace = None
    gspec = plt.GridSpec(n_rols, n_cols, fig, hspace=hspace, wspace=wspace)

    outline_width = (0.3, 0.05)
    size = 300
    bg_width, gap_width = outline_width
    point = np.sqrt(size)

    gap_size = (point + (point * gap_width) * 2) ** 2
    bg_size = (np.sqrt(gap_size) + (point * bg_width) * 2) ** 2

    #    plt.legend(frameon=False)

    is_3d = False

    # if is_3d:
    #     ax = fig.add_subplot(1,1,1,projection='3d')
    # else:
    #     ax = fig.add_subplot(1,1,1)

    axs = []
    for i, gs in enumerate(gspec):
        ax = plt.subplot(gs)

        n = 0.3
        ax.scatter(
            df[x], df[y],
            c=df[df_time_key],
            s=size,
            alpha=0.7 * n,
            marker='X',
            linewidths=0,
            edgecolors=None,
            cmap=cmap
        )

        for trajectory in np.transpose(trajectories, axes=(1, 0, 2)):
            plt.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.3, color='Black')

        states = sorted(df[df_time_key].unique())
        points = np.concatenate(generated, axis=0)
        n_gen = int(points.shape[0] / len(states))
        colors = [state for state in states for i in range(n_gen)]
        print(colors)
        n = 1
        o = '.'
        ax.scatter(
            points[:, 0], points[:, 1],
            c='black',
            s=bg_size,
            alpha=1 * n,
            marker=o,
            linewidths=0,
            edgecolors=None
        )
        ax.scatter(
            points[:, 0], points[:, 1],
            c='white',
            s=gap_size,
            alpha=1 * n,
            marker=o,
            linewidths=0,
            edgecolors=None
        )
        pnts = ax.scatter(
            points[:, 0], points[:, 1],
            c=colors,
            s=size,
            alpha=0.7 * n,
            marker=o,
            linewidths=0,
            edgecolors=None,
            cmap=cmap
        )
        legend_elements = [
            Line2D(
                [0], [0], marker='o',
                color=cmap((i) / (len(states) - 1)), label=f'T{state}',
                markerfacecolor=cmap((i) / (len(states) - 1)), markersize=15,
            )
            for i, state in enumerate(states)
        ]

        leg = plt.legend(handles=legend_elements, loc='upper left')
        ax.add_artist(leg)

        legend_elements = [
            Line2D(
                [0], [0], marker='X', color='w',
                label='Ground Truth', markerfacecolor=cmap(0), markersize=15, alpha=0.3
            ),
            Line2D([0], [0], marker='o', color='w', label='Predicted', markerfacecolor=cmap(.999), markersize=15),
            Line2D([0], [0], color='black', lw=2, label='Trajectory')

        ]
        leg = plt.legend(handles=legend_elements, loc='upper right')
        ax.add_artist(leg)

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.get_xaxis().get_major_formatter().set_scientific(False)
        ax.get_yaxis().get_major_formatter().set_scientific(False)
        kwargs = dict(bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.tick_params(which="both", **kwargs)
        ax.set_frame_on(False)
        ax.patch.set_alpha(0)

        axs.append(ax)

    if save:
        # NOTE: savefig complains image is too large but saves it anyway.
        try:
            fig.savefig(os.path.expanduser(os.path.join(path, file)))
        except ValueError:
            pass
    plt.show()
    return fig
def plot_comparision(
        df, generated, trajectories,
        palette='viridis', df_time_key='samples',
        save=False, path='.', file='comparision.png',
        x='d1', y='d2', z='d3', is_3d=False
):
    if not os.path.isdir(path):
        os.makedirs(path)

    if not is_3d:
        return new_plot_comparisions(
            df, generated, trajectories,
            palette=palette, df_time_key=df_time_key,
            x=x, y=y, z=z, is_3d=is_3d,
            groups=None,
            save=save, path=path, file=file,
        )

    s = 1
    fig = plt.figure(figsize=(12 / s, 8 / s), dpi=300)
    if is_3d:
        ax = fig.add_subplot(1, 1, 1, projection='3d')
    else:
        ax = fig.add_subplot(1, 1, 1)

    states = sorted(df[df_time_key].unique())

    if is_3d:
        ax.scatter(
            df[x], df[y], df[z],
            cmap=palette, alpha=0.3,
            c=df[df_time_key],
            s=df[df_time_key],
            marker='X',
        )
    else:
        sns.scatterplot(
            data=df, x=x, y=y, palette=palette, alpha=0.3,
            hue=df_time_key, style=df_time_key, size=df_time_key,
            markers={g: 'X' for g in states},
            sizes={g: 100 for g in states},
            ax=ax, legend=False
        )

    if not isinstance(generated, np.ndarray):
        generated = to_np(generated)
    points = np.concatenate(generated, axis=0)
    n_gen = int(points.shape[0] / len(states))
    colors = [state for state in states for i in range(n_gen)]

    if is_3d:
        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            cmap=palette,
            c=colors,
        )
    else:
        sns.scatterplot(
            x=points[:, 0], y=points[:, 1], palette=palette,
            hue=colors,
            ax=ax, legend=False
        )

    ax.legend(title='Timepoint', loc='upper left', labels=['Ground Truth', 'Predicted'])
    ax.set_title('ODE Points compared to Ground Truth')

    if is_3d:
        for trajectory in np.transpose(trajectories, axes=(1, 0, 2)):
            plt.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], alpha=0.1, color='Black');
    else:
        for trajectory in np.transpose(trajectories, axes=(1, 0, 2)):
            plt.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.1, color='Black');

    if save:
        # NOTE: savefig complains image is too large but saves it anyway.
        try:
            fig.savefig(os.path.expanduser(os.path.join(path, file)))
        except ValueError:
            pass
    plt.show()
    return fig
plot_comparision(df, np.stack(preds,axis=0), trajectories_2dim,
    palette = 'viridis', df_time_key='samples',
    save=True, path='.', file='comparision.png',
    x='x1', y='x2', is_3d=False)