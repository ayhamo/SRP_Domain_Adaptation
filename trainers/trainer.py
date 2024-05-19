import torch
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
import warnings
import sklearn.exceptions

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
import collections

from sklearn.metrics import accuracy_score
from dataloader.dataloader import data_generator
from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class
from algorithms.utils import fix_randomness, starting_logs
from algorithms.RAINCOAT import RAINCOAT
from algorithms.utils import AverageMeter
from sklearn.metrics import f1_score

torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


class cross_domain_trainer(object):
    """
    This class contain the main training functions for our AdAtime
    """

    def __init__(self, args):
        self.da_method = args.da_method  # Selected  DA Method
        self.dataset = args.dataset  # Selected  Dataset
        self.backbone = args.backbone
        self.device = torch.device(args.device)  # device

        self.run_description = args.run_description
        self.experiment_description = args.experiment_description

        self.best_f1 = 0
        # paths
        self.home_path = os.getcwd()
        self.save_dir = args.save_dir
        self.data_path = os.path.join(args.data_path, self.dataset)
        self.create_save_dir()

        # Specify runs
        self.num_runs = args.num_runs

        # get dataset and base model configs
        self.dataset_configs, self.hparams_class = self.get_configs()

        # to fix dimension of features in classifier and discriminator networks.
        # self.dataset_configs.final_out_channels = self.dataset_configs.tcn_final_out_channles if args.backbone == "TCN" else self.dataset_configs.final_out_channels
        self.dataset_configs.final_out_channels = (
            self.dataset_configs.final_out_channels
        )

        # Specify number of hparams
        self.default_hparams = {
            **self.hparams_class.alg_hparams[self.da_method],
            **self.hparams_class.train_params,
        }

    def train(self):

        run_name = f"{self.run_description}"
        self.hparams = self.default_hparams
        # Logging
        self.exp_log_dir = os.path.join(
            self.save_dir, self.experiment_description, run_name
        )
        os.makedirs(self.exp_log_dir, exist_ok=True)

        scenarios = (
            self.dataset_configs.scenarios
        )  # return the scenarios given a specific dataset.
        df_a = pd.DataFrame(columns=["scenario", "run_id", "accuracy", "f1", "H-score"])
        self.trg_acc_list = []
        for i in scenarios:
            src_id = i[0]
            trg_id = i[1]
            for run_id in range(self.num_runs):  # specify number of consecutive runs
                # fixing random seed
                fix_randomness(run_id)

                # Logging
                self.logger, self.scenario_log_dir = starting_logs(
                    self.dataset,
                    self.da_method,
                    self.exp_log_dir,
                    src_id,
                    trg_id,
                    run_id,
                )
                self.fpath = os.path.join(self.home_path, self.scenario_log_dir, "backbone.pth")
                self.cpath = os.path.join(self.home_path, self.scenario_log_dir, "classifier.pth")
                
                self.best_f1 = 0
                # Load data
                self.load_data(src_id, trg_id)

                # get algorithm
                algorithm = RAINCOAT(self.dataset_configs, self.hparams, self.device)
                algorithm.to(self.device)
                self.algorithm = algorithm

                # TODO Add losses to graph
                loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                # training..
                for epoch in range(1, self.hparams["num_epochs"] + 1):
                    joint_loaders = zip(self.src_train_dl, self.trg_train_dl)

                    algorithm.train()

                    # for loop is defiend becuase some senarios have more than 1 batch and some only have 1 batch
                    for (src_x, src_y), (trg_x, _) in joint_loaders:
                        src_x, src_y, trg_x = (
                            src_x.float().to(self.device),
                            src_y.long().to(self.device),
                            trg_x.float().to(self.device),
                        )

                        losses = algorithm.update(src_x, src_y, trg_x)

                        for key, val in losses.items():
                            loss_avg_meters[key].update(val, src_x.size(0))

                    # logging
                    self.logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
                    acc, f1 = self.eval()

                    if f1 > self.best_f1:
                        self.best_f1 = f1
                        self.logger.debug(f"best f1: {self.best_f1}")
                        torch.save(self.algorithm.feature_extractor.state_dict(), self.fpath)
                        torch.save(self.algorithm.classifier.state_dict(), self.cpath)

                self.logger.debug("===== Correct ====")
                for epoch in range(1, self.hparams["num_epochs"] + 1):
                    joint_loaders = zip(self.src_train_dl, self.trg_train_dl)
                    algorithm.train()
                    for (src_x, src_y), (trg_x, _) in joint_loaders:
                        src_x, src_y, trg_x = (
                        src_x.float().to(self.device),
                        src_y.long().to(self.device),
                        trg_x.float().to(self.device),)

                        algorithm.correct(src_x, src_y, trg_x)

                acc, f1 = self.eval()

                if f1 >= self.best_f1:
                    self.best_f1 = f1
                    self.logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
                    self.logger.debug(f"best f1: {self.best_f1}")
                    torch.save(self.algorithm.feature_extractor.state_dict(), self.fpath)
                    torch.save(self.algorithm.classifier.state_dict(), self.cpath)

                acc, f1 = self.eval(final=True)
                log = {"scenario": i, "run_id": run_id, "accuracy": acc, "f1": f1}
                new_row = pd.DataFrame([log])
                df_a = pd.concat([df_a, new_row], ignore_index=True)

        mean_acc, std_acc, mean_f1, std_f1 = self.avg_result(df_a)
        log = {
            "scenario": "Avg Accuracy: " + str(mean_acc),
            "run_id": "Accuracy STD: " + str(std_acc),
            "accuracy": "Avg F1: " + str(mean_f1),
            "f1": "F1 STD: " + str(std_f1),
        }

        new_row = pd.DataFrame([log])
        df_a = pd.concat([df_a, new_row], ignore_index=True)
        path = os.path.join(self.exp_log_dir, "average_correct.csv")
        df_a.to_csv(path, sep=",", index=False)

    def visualize(self):
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        feature_extractor.eval()
        
        self.trg_true_labels = np.array([])
        self.trg_all_features = []

        self.src_true_labels = np.array([])
        self.src_all_features = []

        with torch.no_grad():

            # for data, labels in self.trg_test_dl:
            for data, labels in self.trg_train_dl:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)
                features, _ = feature_extractor(data)

                self.trg_all_features.append(features.cpu().numpy())
                self.trg_true_labels = np.append(
                    self.trg_true_labels, labels.data.cpu().numpy()
                )

            for data, labels in self.src_train_dl:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)
                features, _ = feature_extractor(data)

                self.src_all_features.append(features.cpu().numpy())
                self.src_true_labels = np.append(
                    self.src_true_labels, labels.data.cpu().numpy()
                )
            self.src_all_features = np.vstack(self.src_all_features)
            self.trg_all_features = np.vstack(self.trg_all_features)

    def eval(self, final=False):
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)
        if final == True:
            feature_extractor.load_state_dict(torch.load(self.fpath))
            classifier.load_state_dict(torch.load(self.cpath))
        feature_extractor.eval()
        classifier.eval()

        total_loss_ = []

        self.trg_pred_labels = np.array([])
        self.trg_true_labels = np.array([])

        with torch.no_grad():
            for data, labels in self.trg_test_dl:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features, _ = feature_extractor(data)
                predictions = classifier(features)

                # compute loss
                loss = F.cross_entropy(predictions, labels)
                total_loss_.append(loss.item())
                pred = predictions.detach().argmax(
                    dim=1
                )  # get the index of the max log-probability

                self.trg_pred_labels = np.append(
                    self.trg_pred_labels, pred.cpu().numpy()
                )
                self.trg_true_labels = np.append(
                    self.trg_true_labels, labels.data.cpu().numpy()
                )
        accuracy = accuracy_score(self.trg_true_labels, self.trg_pred_labels)
        f1 = f1_score(
            self.trg_pred_labels, self.trg_true_labels, pos_label=None, average="macro"
        )
        return accuracy * 100, f1

    def get_configs(self):
        dataset_class = get_dataset_class(self.dataset)
        hparams_class = get_hparams_class(self.dataset)
        return dataset_class(), hparams_class()

    def load_data(self, src_id, trg_id):
        self.src_train_dl, self.src_test_dl = data_generator(
            self.data_path, src_id, self.dataset_configs, self.hparams
        )
        self.trg_train_dl, self.trg_test_dl = data_generator(
            self.data_path, trg_id, self.dataset_configs, self.hparams
        )

    def create_save_dir(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def avg_result(self, df):
        mean_acc = df.groupby("scenario", as_index=False, sort=False)["accuracy"].mean()
        mean_f1 = df.groupby("scenario", as_index=False, sort=False)["f1"].mean()
        std_acc = df.groupby("run_id", as_index=False, sort=False)["accuracy"].mean()
        std_f1 = df.groupby("run_id", as_index=False, sort=False)["f1"].mean()

        return (
            mean_acc["accuracy"].mean(),
            std_acc["accuracy"].std(),
            mean_f1["f1"].mean(),
            std_f1["f1"].std(),
        )
