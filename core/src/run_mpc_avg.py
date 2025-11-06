import os
import shutil
import random
from argparse import Namespace, ArgumentParser
import yaml
import numpy as np
import torch

# 1. import our new experiment class
from allocate.experiment_mpc_avg import MpcAvgExperiment

def run_mpc_avg_experiment(configs):
    # Fix random seed so my results are the same every time
    random.seed(configs.random_seed)
    np.random.seed(configs.random_seed)
    torch.manual_seed(configs.random_seed)

    # 2. make the experiment object
    experiment = MpcAvgExperiment(configs)

    # 3. Evaluate!!
    print(f'{">" * 20} {"Start Averaged MPC testing:":<15} {configs.exp_id} {"<" * 20}')
    
    # 4. just call the new evaluate func
    metrics = experiment.evaluate()
    
    print(f'Average MPC Regret: {metrics["Avg_MPC_Regret"]:.8f}')

if __name__ == '__main__':
    # this is the standard python entrypoint
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    # load the yaml file
    with open(args.config, 'r') as fin:
        configs = yaml.safe_load(fin)

    # make the output folder
    exp_dir = os.path.join('output', configs['Experiment']['exp_id'])
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # copy the config file so i dont forget what i ran
    config_path = os.path.join(exp_dir, 'mpc_avg_config.yaml')
    with open(config_path, 'w') as fout:
        yaml.dump(configs, fout, indent=4, sort_keys=False)

    # this namespace thing is just a way to access the config 
    # like configs.model instead of configs['Model']['model']
    configs = Namespace(**{
        arg: val
        for _, args in configs.items()
        for arg, val in args.items()
    })

    run_mpc_avg_experiment(configs)