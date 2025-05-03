import json
import random
import seaborn as sns
from src.fit_epoch import get_ROCKET
sns.color_palette("bright")
import os
from types import SimpleNamespace
import numpy as np
import torch
from joblib import load
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from src.config_Veres import load_data
from src.model import ForwardSDE
interval = 20
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

seed_everything(37)
path = "RESULTS/Veres/alltime/config.pt"
config = SimpleNamespace(**torch.load(path))
x, y, c, config = load_data(config)

model = ForwardSDE(config).to(config.device)
checkpoint = torch.load('RESULTS/Veres/alltime/train.best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
gumbel_tau = checkpoint['gumbel_tau']
model.eval()
#######################################################################################################################
ts = [0] + config.train_t
y_ts = [np.float64(y[ts_i]) for ts_i in ts]
x_i = x[0]
x_r_i = x_i.to(config.device)
sample_time = np.linspace(np.min(y_ts), np.max(y_ts), num=interval)
sample_time = [np.float64(ts_i) for ts_i in sample_time]
trajectories, _, _ = model(sample_time, x_r_i, gumbel_tau)
#######################################################
roc_model = torch.load(f'data/models/Veres_ROCKET_0_n_kernels=250.pt')
Logistic = torch.jit.load(f"data/models/Veres_Logistic Regression0_n_kernels=250.pt").float()
######################################################################################
if not os.path.exists('x_r_s_2D.npy') or not os.path.exists('x_clc_2D.npy'):
    x_clc = np.array([Logistic.predict_proba(get_ROCKET(roc_model, trajectories[i, :, :].view(-1, trajectories.shape[2]))).detach().numpy() for i in
                      range(trajectories.shape[0])])
    np.save('x_clc_2D.npy', x_clc)
    um = load(r'data/data/Veres/alltime/um_transformer.joblib')
    x_r_s = np.array([um.transform(trajectories[i].cpu().detach().numpy()) for i in range(trajectories.shape[0])])
    np.save('x_r_s_2D.npy', x_r_s)
else:
    x_r_s = np.load('x_r_s_2D.npy')
    x_clc = np.load('x_clc_2D.npy')

num_timesteps, num_samples, num_coords = x_r_s.shape
with open('cell_list.json', 'r') as file:
    class_labels = json.load(file)
print(class_labels)
step=0.5
fig, ax = plt.subplots(figsize=(15, 10))
ax.set_xlim(x_r_s[:, :, 0].min()-step, x_r_s[:, :, 0].max()+step)
ax.set_ylim(x_r_s[:, :, 1].min()-step, x_r_s[:, :, 1].max()+step)
ax.set_facecolor('white')
fig.patch.set_facecolor('white')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')
for spine in ax.spines.values():
    spine.set_visible(False)
# ax.set_title("Trajectory Evolution with Classification")

outline_width = (0.3, 0.05)
size = 300
bg_width, gap_width = outline_width
point = np.sqrt(size)
gap_size = (point + (point * gap_width) * 2) ** 2
bg_size = (np.sqrt(gap_size) + (point * bg_width) * 2) ** 2
n = 0.9
o = '.'

scat1 = ax.scatter([], [], c='black', s=bg_size, alpha=1 * n, marker=o, linewidths=0)
scat2 = ax.scatter([], [], c='white', s=gap_size, alpha=1 * n, marker=o, linewidths=0)
time_text = ax.text(
    0.02, 0.95, "",
    transform=ax.transAxes,
    fontsize=12,
    color='white',
    fontweight='bold',
    fontstyle='italic',
    bbox=dict(facecolor='#1a1f3d', alpha=0.75, edgecolor='none')
)
colors = ["#f43b7e", "#344158", "#3899ae", "#C82423", "#f1d496", "#4eb7ac",
          "#ffb3e6", "#f0d4c1", "#cb4b54", "#9873B9", "#14517C", "#b3c7db"]
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'P', 'X']
scatters = [
    ax.scatter([], [], color=colors[i], alpha=0.3, label=class_labels[i], marker=markers[i])
    for i in range(len(class_labels))]

def update(frame):

    current_points = x_r_s[frame, :, :]
    scat1.set_offsets(current_points)
    scat2.set_offsets(current_points)

    current_class = x_clc[frame, :, :]
    class_indices = np.argmax(current_class, axis=1)

    for i in range(len(class_labels)):
        mask = (class_indices == i)
        scatters[i].set_offsets(current_points[mask])

    time_text.set_text(f"Time: {sample_time[frame]:.2f}")
    return [scat1, scat2] + scatters + [time_text]

ani = FuncAnimation(fig, update, frames=num_timesteps, interval=interval, blit=True)
handles = [plt.Line2D([0], [0], marker=markers[i], color='w', markerfacecolor=colors[i], markersize=10) for i in range(len(class_labels))]
ax.legend(handles, class_labels, bbox_to_anchor=(1.17, 1), loc="upper right", frameon=False)
plt.tight_layout()
ani.save("Veres.gif", writer="imagemagick", dpi=800, fps=144)
# ani.save("MH.mp4", writer='ffmpeg', dpi=800, fps=144)
