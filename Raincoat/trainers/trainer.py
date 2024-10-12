import torch
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np

import multiprocessing
from functools import partial

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from dataloader.dataloader import data_generator
from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class
from algorithms.utils import fix_randomness, starting_logs
from algorithms.RAINCOAT import RAINCOAT

torch.backends.cudnn.benchmark = True


class cross_domain_trainer(object):

    def __init__(self, args):
        self.dataset = args.dataset  # Selected  Dataset
        self.device = torch.device(args.device)  # device
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

        # Specify number of hparams
        self.default_hparams = {
            **self.hparams_class.train_params,
        }


    def single_run_scenario(self, scenario, run_id):
        src_id, trg_id = scenario
        
        # fixing random seed
        fix_randomness(run_id)

        # Logging
        self.logger, self.scenario_log_dir = starting_logs(
                    self.dataset,
                    "RAINCOAT",
                    self.exp_log_dir,
                    src_id,
                    trg_id,
                    run_id,
                )
        self.fpath = os.path.join(self.home_path, self.scenario_log_dir, "backbone.pth")
        self.cpath = os.path.join(self.home_path, self.scenario_log_dir, "classifier.pth")
        
        best_f1 = 0

        # Load data
        self.load_data(src_id, trg_id)

        # get algorithm
        algorithm = RAINCOAT(self.dataset_configs, self.hparams, self.device)
        algorithm.to(self.device)
        self.algorithm = algorithm

        self.best_feature_extractor_state = None
        self.best_classifier_state = None
        
        # training
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
                losses = algorithm.align(src_x, src_y, trg_x)

            acc, f1 = self.eval(self.trg_val_dl)

            if f1 > best_f1:
                self.logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
                best_f1 = f1
                self.logger.debug(f"best f1: {best_f1}")

                algorithm.scheduler.step()

                self.best_feature_extractor_state = algorithm.feature_extractor.state_dict()
                self.best_classifier_state = algorithm.classifier.state_dict()

        self.logger.debug("===== Correct ====")
        for epoch in range(1, self.hparams["corr_epochs"] + 1):
            joint_loaders = zip(self.src_train_dl, self.trg_train_dl)
            algorithm.train()

            for (src_x, src_y), (trg_x, _) in joint_loaders:
                src_x, src_y, trg_x = (
                    src_x.float().to(self.device),
                    src_y.long().to(self.device),
                    trg_x.float().to(self.device),
                )
                correct_losses = algorithm.correct(src_x, src_y, trg_x)

            acc, f1 = self.eval(self.trg_val_dl)

            if f1 >= best_f1:
                self.logger.debug(f'[Epoch : {epoch}/{self.hparams["corr_epochs"]}]')
                best_f1 = f1
                self.logger.debug(f"best f1: {best_f1}")

                algorithm.coscheduler.step()

                self.best_feature_extractor_state = algorithm.feature_extractor.state_dict()
                self.best_classifier_state = algorithm.classifier.state_dict()

        # to save file only once, not at each best f1
        # current behavoiur is not to save model, and use variables
        #torch.save(best_feature_extractor_state, self.fpath)
        #torch.save(best_classifier_state, self.cpath)
        
        # at final eval, we use test now
        acc, f1 = self.eval(self.trg_test_dl, final=True)

        return {"scenario": scenario, "run_id": run_id, "accuracy": acc, "f1": f1}

    def train(self):
        self.hparams = self.default_hparams

        # Logging
        self.exp_log_dir = os.path.join(self.save_dir, f"RAINCOAT ClosedSet", self.experiment_description)
        os.makedirs(self.exp_log_dir, exist_ok=True)

        scenarios = self.dataset_configs.scenarios
        df_a = pd.DataFrame(columns=["scenario", "run_id", "accuracy", "f1"])

        self.trg_acc_list = []

        for scenario in scenarios:
            # Create a partial function with fixed scenario
            run_iteration = partial(self.single_run_scenario, scenario)
            
            # Create a pool of workers
            with multiprocessing.Pool(processes=self.num_runs) as pool:
                # Run the iterations in parallel
                results = pool.map(run_iteration, range(self.num_runs))

            # Process and log results
            for result in results:
                log = {
                    "scenario": result["scenario"],
                    "run_id": result["run_id"],
                    "accuracy": result["accuracy"],
                    "f1": result["f1"]
                }
                new_row = pd.DataFrame([log])
                df_a = pd.concat([df_a, new_row], ignore_index=True)
                self.trg_acc_list.append(result['accuracy'])

        # Calculate average results
        mean_acc, std_acc, mean_f1, std_f1 = self.avg_result(df_a)
        log = {
            "scenario": "Avg Accuracy: " + str(mean_acc),
            "run_id": "Accuracy STD: " + str(std_acc),
            "accuracy": "Avg F1: " + str(mean_f1),
            "f1": "F1 STD: " + str(std_f1),
        }

        new_row = pd.DataFrame([log])
        df_a = pd.concat([df_a, new_row], ignore_index=True)

        # Save results to CSV file
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

    def eval(self, dataloader, final=False):
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)
        
        if final == True:
            # current behavoiur to not to save models and use varaibles
            # at the end we might want to change this to save the best models for each secanrio

            #feature_extractor.load_state_dict(torch.load(self.fpath))
            #classifier.load_state_dict(torch.load(self.cpath))
            
            self.algorithm.feature_extractor.load_state_dict(self.best_feature_extractor_state)
            self.algorithm.classifier.load_state_dict(self.best_classifier_state)
        feature_extractor.eval()
        classifier.eval()

        total_loss_ = []

        self.trg_pred_labels = np.array([])
        self.trg_true_labels = np.array([])

        with torch.no_grad():
            # now using datalaoder instead of only test
            for data, labels in dataloader:
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
        self.src_train_dl, self.src_val_dl, self.src_test_dl = data_generator(
                self.data_path, src_id, self.dataset_configs, self.hparams)
        self.trg_train_dl, self.trg_val_dl, self.trg_test_dl = data_generator(
                self.data_path, trg_id, self.dataset_configs, self.hparams)
        

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
