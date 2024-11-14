from functools import partial
import multiprocessing
import torch
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np

import copy
import diptest
from sklearn.cluster import KMeans
from algorithms.RAINCOAT import RAINCOAT
from sklearn.metrics import accuracy_score
from dataloader.uni_dataloader import data_generator
from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class
from algorithms.utils import fix_randomness, starting_logs
from sklearn.metrics import f1_score
torch.backends.cudnn.benchmark = True

import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)
np.seterr(all="ignore")

class cross_domain_trainer(object):

    def __init__(self, args):
        self.dataset = args.dataset  # Selected  Dataset
        self.device = torch.device(args.device)  # device
        self.experiment_description = args.experiment_description

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
        self.default_hparams = {**self.hparams_class.train_params}

    def train(self):
        self.hparams = self.default_hparams

        # Logging
        self.exp_log_dir = os.path.join(self.save_dir, "RAINCOAT UNIDA", self.experiment_description)
        os.makedirs(self.exp_log_dir, exist_ok=True)

        scenarios = self.dataset_configs.H_scenarios  # Get training scenarios
        df_c = pd.DataFrame(columns=['scenario', 'run_id', 'accuracy', 'f1', 'H-score'])

        # to fix cuda init error
        multiprocessing.set_start_method('spawn')

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
                    "f1": result["f1"],
                    "H-score": result["H-score"]
                }
                new_row = pd.DataFrame([log])
                df_c = pd.concat([df_c, new_row], ignore_index=True)

        # Calculate average results
        mean_acc, std_acc, mean_f1, std_f1, mean_H, std_H = self.avg_result(df_c)
        log = {
            "scenario": "Avg Accuracy: " + str(mean_acc),
            "run_id": "Accuracy STD: " + str(std_acc),
            "accuracy": "Avg F1: " + str(mean_f1),
            "f1": "F1 STD: " + str(std_f1),
            "H-score": "H-score " + str(mean_H) + " H-STD " + str(std_H),
        }

        new_row = pd.DataFrame([log])
        df_c = pd.concat([df_c, new_row], ignore_index=True)

        # Save results to CSV file
        path = os.path.join(self.exp_log_dir, "average_correct.csv")
        df_c.to_csv(path, sep=",", index=False)


    def single_run_scenario(self, scenario, run_id):
        src_id, trg_id = scenario

        # device based on run_id
        #self.device = self.get_device(run_id)

        # fixing random seed
        fix_randomness(run_id)

        # Logging
        self.logger, self.scenario_log_dir = starting_logs(
                    self.dataset,
                    "RAINCOAT", 
                    self.exp_log_dir,
                    src_id, 
                    trg_id, 
                    run_id)
        
        self.fpath = os.path.join(self.home_path, self.scenario_log_dir, 'backbone.pth')
        self.cpath = os.path.join(self.home_path, self.scenario_log_dir, 'classifier.pth')

        best_f1 = 0

        # Load data
        self.load_data(src_id, trg_id)

        # get algorithm
        algorithm = RAINCOAT(self.dataset_configs, self.hparams, self.device)
        algorithm.to(self.device)
        self.algorithm = algorithm

        self.best_feature_extractor_state = None
        self.best_classifier_state = None

        # identifies private classes and masks them as "-1" in the target labels
        tar_uni_label_train, pri_class = self.preprocess_labels(self.src_train_dl, self.trg_train_dl)
        tar_uni_label_test, pri_class = self.preprocess_labels(self.src_train_dl, self.trg_test_dl)
        size_ltrain, size_ltest = len(tar_uni_label_train), len(tar_uni_label_test)

        # training
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            joint_loaders = zip(self.src_train_dl, self.trg_train_dl)
            algorithm.train()

            for (src_x, src_y, _), (trg_x, _, trg_index) in joint_loaders:
                src_x, src_y, trg_x, trg_index = src_x.float().to(self.device), src_y.long().to(self.device), \
                                                    trg_x.float().to(self.device), trg_index.to(self.device)

                losses = algorithm.align(src_x, src_y, trg_x)
            
            # for now, target and validation are the same becuase random_split gives a set
            # that breaks the code
            acc, f1, _ = self.eval(self.trg_val_dl, get_H= False)

            if f1 > best_f1:
                self.logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
                best_f1 = f1
                self.logger.debug(f"best f1: {best_f1}")

                algorithm.scheduler.step()

                # Save best model states
                self.best_feature_extractor_state = self.algorithm.feature_extractor.state_dict()
                self.best_classifier_state = self.algorithm.classifier.state_dict()

        print("===== Correct ====")
        # Explained below what is this
        dis2proto_a = self.calc_distance(size_ltrain, self.trg_train_dl)
        dis2proto_a_test = self.calc_distance(size_ltest, self.trg_test_dl)

        for epoch in range(1, self.hparams["corr_epochs"] + 1):
            joint_loaders = zip(self.src_train_dl, self.trg_train_dl)
            algorithm.train()

            for (src_x, src_y, _), (trg_x, _, trg_index) in joint_loaders:
                src_x, src_y, trg_x, trg_index = src_x.float().to(self.device), src_y.long().to(self.device), \
                                                    trg_x.float().to(self.device), trg_index.to(self.device)

                correct_losses = algorithm.correct(src_x, src_y, trg_x)
            
            acc, f1, _ = self.eval(self.trg_val_dl, get_H= False)

            if f1 > best_f1:
                self.logger.debug(f'[Epoch : {epoch}/{self.hparams["corr_epochs"]}]')
                best_f1 = f1
                self.logger.debug(f"best f1: {best_f1}")

                algorithm.coscheduler.step()

                # Save best model states
                self.best_feature_extractor_state = self.algorithm.feature_extractor.state_dict()
                self.best_classifier_state = self.algorithm.classifier.state_dict()

        # to save file only once, not at each best f1
        # we save the state dict that has best model, not last one!
        #torch.save(best_feature_extractor_state, self.fpath)
        #torch.save(best_classifier_state, self.cpath)

        '''
        The issue of H score starts here, where c list would be filled with 1's where it's supposed to contain 
        thresholds for each class to separate private samples based on feature drift. so a c_list 
        with 1's disables private sample detection. 
        
        there is 2 problems here:
        1) the dip < 0.05 condition in learn_t function is never met due to
        "small" Drift where RAINCOAT UniDA assumes "correction" step causes features of private classes to 
        drift "significantly" while those of common classes remain relatively stable. 
        If the drift is too small for all classes, the dip test will not detect bimodality

        2) even if we able to pass dip test, k means that outputs c value (min thresold for drift) for 
        it to be consediered a private class might be just too big, so when we do the condition
        diff > c in detect_private, it's never really statsfied, thus we get 0 or really low value

        we made sure that calc_distance does infact have diffrenet values, ruling it out
        '''
        
        # cosine similarity between the features of each sample in the given dataloader 
        # and the prototypes of the predicted class
        dis2proto_c = self.calc_distance(size_ltrain, self.trg_train_dl)
        dis2proto_c_test = self.calc_distance(size_ltest, self.trg_test_dl)

        # thresholds (c_list) used in detect_private to distinguish between 
        # known and unknown classes based on their feature drift, that is a value greater than c_value drift
        c_list = self.learn_t(dis2proto_a, dis2proto_c)
        print(f"threshold list: {c_list}")

        # This is basiclly removing the original labels with ones that are masked by pre-prcoess
        # This is done to work in H_score method, bad way to do so!, but does not make an issue
        self.trg_true_labels = tar_uni_label_test
                                         
        acc, f1, H = self.detect_private(self.trg_test_dl, dis2proto_a_test, dis2proto_c_test, tar_uni_label_test, c_list)

        '''
        -Source 3 -> Target 2:
        All classes pass the dip test (c list dip worked).
        H-score is good (0.35). This scenario seems to be working as intended.

        -Source 3 -> Target 7:
        All classes pass the dip test (c list dip worked).
        Problem: H-score is 0, i suspect this due to kmean clsuter failing to cluster it, leading to bad to mask
        
        -Source 13 -> Target 15:
        Class 2 fails the dip test ("Failed cc shape"). fewer than 4 datapoints in the class (no issue here)
        Problem: H-score is 0, i suspect this due to kmean clsuter failing to cluster it, leading to bad to mask
        
        -Source 14 -> Target 19:
        All classes pass the dip test (c list dip worked).
        Problem: H-score is 0. The same issue as in previous.
        
        -Source 27 -> Target 28:
        All classes pass the dip test.
        Thresholds appear reasonable.
        H-score is non-zero (0.165).
        
        Source 1 -> Target 0:
        Classes 2 and 3 fail the dip test ("Failed cc shape").
        Problem: H-score is not zero (0.239), so this is ok

        Source 1 -> Target 3:
        Classes 1 and 3 fail the dip test ("Failed cc shape")
        Failed to detect private here in target in a class(dip target failed!), other one worked but
        Problem: H-score is 0. kmeans most prob
        
        Source 10 -> Target 11:
        All classes pass the dip test.
        1 class failed detection, 2 worked but
        Problem: H-score is 0. kmeans again
        
        Source 22 -> Target 17:
        All classes pass the dip test.
        1 class failed detection, 2 worked but
        Problem: H-score is 0.
        
        Source 27 -> Target 15:
        Class 1 fails the the cc shape
        found private but
        Problem: H-score is 0.
        '''

        return {'scenario': scenario, 'run_id': run_id, 'accuracy': acc, 'f1': f1, 'H-score': H}


    def preprocess_labels(self, source_loader, target_loader):
        trg_y = copy.deepcopy(target_loader.dataset.y_data)
        src_y = source_loader.dataset.y_data
        pri_c = np.setdiff1d(trg_y, src_y)
        mask = np.isin(trg_y, pri_c)
        trg_y[mask] = -1
        return trg_y, pri_c

    def detect_private(self, datalaoder, d1, d2, tar_uni_label, c_list):
        
        # in here, we refactor eval to only give us preds of target dataset
        self.trg_pred_labels = self.eval(datalaoder, get_preds = True)

        # now we see if we have private
        diff = np.abs(d2-d1)
        # TODO this 6 shuld be wrong?
        for i in range(6):
            cat = np.where(self.trg_pred_labels==i)
            cc = diff[cat]
            if cc.shape[0]>3:
                dip, pval = diptest.diptest(diff[cat])
                # this was 0.05 (5% error), we changed to 10%
                if dip < 0.10:
                    print(f"contain private in target with dip: {dip}")
                    c = c_list[i]
                    m1 = np.where(diff>c)
                    m2 = np.where(self.trg_pred_labels==i)
                    mask = np.intersect1d(m1, m2)
                    # print(m1, m2, mask)
                    self.trg_pred_labels[mask] = -1
                else:
                    print(f"detect private target dip Failed: {dip}")

        
        # current behavoiur to not to save models and use varaibles
        # at the end we might want to change this to save the best models for each secanrio
        #feature_extractor.load_state_dict(torch.load(self.fpath))
        #classifier.load_state_dict(torch.load(self.cpath))

        self.algorithm.feature_extractor.load_state_dict(self.best_feature_extractor_state)
        self.algorithm.classifier.load_state_dict(self.best_classifier_state)
        
        # No need to change to eval, we have pre-prcoessed with mask as first argument
        # and the predictions we got 
        accuracy = accuracy_score(tar_uni_label, self.trg_pred_labels)
        f1 = f1_score(self.trg_pred_labels, tar_uni_label, pos_label=None, average="macro")

        return accuracy*100, f1, self.H_score()

    def learn_t(self, d1, d2):
        diff = np.abs(d2 - d1)
        c_list = []
        for i in range(6):
            cat = np.where(self.trg_train_dl.dataset.y_data == i)
            cc = diff[cat]
            # check if there is more than 3 datapoints
            if cc.shape[0] > 3:
                dip, pval = diptest.diptest(diff[cat])
                # here is not enough evidence to reject the null hypothesis of unimodality at the 5% error level. 
                # ie the data does not show strong evidence of being multimodal
                # TODO but what is dip(direct measure of unimodality) vs pval(probabilistic measure of unimodality)?
                # this was 0.05 (5% error), we changed to 10%
                if dip < 0.10:
                    print(f"c list dip worked: {dip}")
                    kmeans = KMeans(n_clusters=2, random_state=0, max_iter=5000, n_init=50, init="k-means++").fit(
                        diff[cat].reshape(-1, 1))
                    c = max(kmeans.cluster_centers_)

                else:
                    print(f"c list dip Failed: {dip}")
                    c = 1e10
            else:
                print(f"Failed cc shape")
                c = 1e10
            c_list.append(c)
        return c_list
    
    def calc_distance(self, len_y, dataloader):
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)
        feature_extractor.eval()
        classifier.eval()

        proto = classifier.logits.weight.data

        # normalize the prototype vectors, improve the performance
        #norm = proto.norm(p=2, dim=1, keepdim=True)
        #proto = proto.div(norm.expand_as(norm))

        trg_drift = np.zeros(len_y)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        with torch.no_grad():
            for data, labels, trg_index in dataloader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                features, _ = feature_extractor(data)
                predictions = classifier(features.detach())
                pred_label = torch.argmax(predictions, dim=1)
                proto_M = torch.vstack([proto[l, :] for l in pred_label])
                angle_c = cos(features, proto_M) ** 2
                # the useage of this is unkown
                # dist = (torch.max(predictions,1).values).div(torch.log(angle_c))
                trg_drift[trg_index] = angle_c.cpu().numpy()
        return trg_drift

    def eval(self, dataloader, get_H = False, get_preds = False):
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)
        feature_extractor.eval()
        classifier.eval()

        data = copy.deepcopy(dataloader.dataset.x_data).float().to(self.device)
        labels = dataloader.dataset.y_data.view((-1)).long().to(self.device)

        features, _ = feature_extractor(data)
        predictions = classifier(features)

        pred = predictions.argmax(dim=1)
        pred = pred.cpu().numpy()

        accuracy = accuracy_score(labels.cpu().numpy(), pred)
        f1 = f1_score(pred, labels.cpu().numpy(), pos_label=None, average="macro")

        # we refactor some part of code such that we use eval to return pred
        # of target, that is used during detect_private
        if get_preds:
            return pred

        self.trg_true_labels = labels.cpu().numpy()

        H_score = 0
        if get_H:
            H_score = self.H_score()

        return accuracy * 100, f1, H_score


    def H_score(self):
        class_c = np.where(self.trg_true_labels != -1)
        class_p = np.where(self.trg_true_labels == -1)

        label_c, pred_c = self.trg_true_labels[class_c], self.trg_pred_labels[class_c]
        label_p, pred_p = self.trg_true_labels[class_p], self.trg_pred_labels[class_p]

        acc_c = accuracy_score(label_c, pred_c)
        acc_p = accuracy_score(label_p, pred_p)
        if acc_c == 0 or acc_p == 0:
            H = 0
            print(f"Failed H score: {H}")
        else:
            H = 2 * acc_c * acc_p / (acc_p + acc_c)
            print(f"we have a H score: {H}")
        return H

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
        mean_acc = df.groupby('scenario', as_index=False, sort=False)['accuracy'].mean()
        mean_f1 = df.groupby('scenario', as_index=False, sort=False)['f1'].mean()
        mean_H = df.groupby('scenario', as_index=False, sort=False)['H-score'].mean()

        std_acc = df.groupby('run_id', as_index=False, sort=False)['accuracy'].mean()
        std_f1 = df.groupby('run_id', as_index=False, sort=False)['f1'].mean()
        std_H = df.groupby('run_id', as_index=False, sort=False)['H-score'].mean()

        return (
            mean_acc["accuracy"].mean(),
            std_acc["accuracy"].std(),
            mean_f1["f1"].mean(),
            std_f1["f1"].std(),
            mean_H["H-score"].mean(),
            std_H["H-score"].std()
        )