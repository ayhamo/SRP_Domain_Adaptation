import argparse
from trainers.trainer_uni import cross_domain_trainer

parser = argparse.ArgumentParser()


# ========  Experiments Name ================
parser.add_argument('--save_dir',               default='experiments_logs',         type=str, help='Directory containing all experiments')
parser.add_argument('--experiment_description', default='WISDM',               type=str, help='Name of your experiment (EEG, HAR, HHAR_SA')

# ========= Select the DATASET ==============
parser.add_argument('--data_path',              default=r'./data',                  type=str, help='Path containing dataset')
parser.add_argument('--dataset',                default='WISDM',                      type=str, help='Dataset of choice: (WISDM - EEG - HAR - HHAR_SA, Boiler)')

# ========= Experiment settings ===============
parser.add_argument('--num_runs',               default=1,                          type=int, help='Number of consecutive run with different seeds')
parser.add_argument('--device',                 default='cuda',                   type=str, help='cpu or cuda')

args = parser.parse_args()

if __name__ == "__main__":
    # python main_uni.py --experiment_description WISDM --dataset WISDM --num_runs 1 --device cuda
    trainer = cross_domain_trainer(args)
    trainer.train()
    