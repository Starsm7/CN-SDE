import os
import numpy as np
import torch
from src.config_Veres import config, init_config
import src.fit_epoch as train
from src.evaluation import evaluate_fit
import warnings
warnings.filterwarnings("ignore")
import random
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
seed_everything(967)
leaveouts = [7,2,3,4,5,6,1]
random_numbers = [17]
for random_number in random_numbers:
    for leaveout in leaveouts:
        if os.path.exists(f"RESULTS/Veres/seed{random_number}/leaveout{leaveout}"):
            continue
        seed_everything(random_number)
        args = config()
        args.seed = random_number
        config_train = train.run(args, init_config, leaveouts=[leaveout])
        evaluate_fit(config_train, init_config, use_loss='emd')

        torch.cuda.empty_cache()
        del args, config_train

        print(f"Task with leaveout {leaveout} completed and cleared.")