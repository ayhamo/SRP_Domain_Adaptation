# enter env: env\Scripts\Activate  
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# pip install -r requirement.txt

import argparse
import warnings
from trainers.trainer import cross_domain_trainer
import sklearn.exceptions
import pickle
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

parser = argparse.ArgumentParser()


# ========  Experiments Name ================
parser.add_argument('--save_dir',               default='experiments_logs',         type=str, help='Directory containing all experiments')
parser.add_argument('--experiment_description', default='WISDM',               type=str, help='Name of your experiment (EEG, HAR, HHAR_SA, WISDM')

# ========= Select the DATASET ==============
parser.add_argument('--data_path',              default=r'./data',                  type=str, help='Path containing dataset')
parser.add_argument('--dataset',                default='WISDM',                      type=str, help='Dataset of choice: (WISDM - EEG - HAR - HHAR_SA, Boiler)')

# ========= Experiment settings ===============
parser.add_argument('--num_runs',               default=1,                          type=int, help='Number of consecutive run with different seeds')
parser.add_argument('--device',                 default='cuda',                   type=str, help='cpu or cuda')

args = parser.parse_args()

if __name__ == "__main__":
    # python main.py --experiment_description WISDM --dataset WISDM --num_runs 1
    trainer = cross_domain_trainer(args)
    trainer.train()
    
    #trainer.visualize()
    #dic = {'1':trainer.src_all_features,'2':trainer.src_true_labels,'3':trainer.trg_all_features,'4':trainer.trg_true_labels,'acc': trainer.trg_acc_list}
    #with open('saved_dictionary2.pickle', 'wb') as handle:
    #    pickle.dump(dic, handle)
    