import torch
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
warnings.simplefilter("ignore", category=RuntimeWarning)
np.seterr(all="ignore")
import copy
import diptest
from sklearn.cluster import KMeans
from algorithms.RAINCOAT import RAINCOAT
from sklearn.metrics import accuracy_score
from dataloader.uni_dataloader import data_generator
from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class
from algorithms.utils import fix_randomness, starting_logs
from algorithms.algorithms import get_algorithm_class
from models.models import get_backbone_class
from sklearn.metrics import f1_score
torch.backends.cudnn.benchmark = True  
from sklearn.mixture import GaussianMixture
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)        

class cross_domain_trainer(object):
    """
   This class contain the main training functions for our AdAtime
    """
    def __init__(self, args):
        self.da_method = args.da_method  # Selected  DA Method
        self.dataset = args.dataset  # Selected  Dataset
        self.backbone = args.backbone
        self.device = torch.device(args.device)  # device

        self.run_description = args.experiment_description
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
        self.dataset_configs.final_out_channels = self.dataset_configs.tcn_final_out_channles if args.backbone == "TCN" else self.dataset_configs.final_out_channels

        # Specify number of hparams
        self.default_hparams = {**self.hparams_class.alg_hparams[self.da_method],
                                **self.hparams_class.train_params}


    def train(self):

        run_name = f"{self.run_description}"
        self.hparams = self.default_hparams
        # Logging
        self.exp_log_dir = os.path.join(self.save_dir, self.experiment_description, run_name)
        os.makedirs(self.exp_log_dir, exist_ok=True)

        scenarios = self.dataset_configs.scenarios  # return the scenarios given a specific dataset.
        df_c = pd.DataFrame(columns=['scenario','run_id','accuracy','f1','H-score'])
        for i in scenarios:
            src_id = i[0]
            trg_id = i[1]
            for run_id in range(self.num_runs):  # specify number of consecutive runs
                # fixing random seed
                fix_randomness(run_id)

                self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.exp_log_dir,
                                                                   src_id, trg_id, run_id)
                self.fpath = os.path.join(self.home_path, self.scenario_log_dir, 'backbone.pth')
                self.cpath = os.path.join(self.home_path, self.scenario_log_dir, 'classifier.pth')
                
                self.best_f1 = 0
                # Load data
                self.load_data(src_id, trg_id)

                algorithm = RAINCOAT(self.dataset_configs, self.hparams, self.device)
                algorithm.to(self.device)
                self.algorithm = algorithm

                tar_uni_label_train, pri_class = self.preprocess_labels(self.src_train_dl, self.trg_train_dl)
                tar_uni_label_test, pri_class = self.preprocess_labels(self.src_train_dl, self.trg_test_dl)
                size_ltrain, size_ltest = len(tar_uni_label_train),len(tar_uni_label_test)

                #training
                for epoch in range(1, self.hparams["num_epochs"] + 1):
                    joint_loaders = zip(self.src_train_dl, self.trg_train_dl)

                    algorithm.train()
                
                    for (src_x, src_y, _), (trg_x, _, trg_index) in joint_loaders:
                        src_x, src_y, trg_x, trg_index = src_x.float().to(self.device), src_y.long().to(self.device), \
                                              trg_x.float().to(self.device), trg_index.to(self.device)
                        
                        algorithm.update(src_x, src_y, trg_x)
                    
                    # logging
                    self.logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
                    acc, f1, _ = self.evaluate_RAINCOAT(self.trg_test_dl.dataset.y_data)

                    if f1 > self.best_f1:
                        self.best_f1 = f1
                        self.logger.debug(f"best f1: {self.best_f1}")
                        #torch.save(self.algorithm.feature_extractor.state_dict(), self.fpath)
                        #torch.save(self.algorithm.classifier.state_dict(), self.cpath)
                    
                # Step 2: correct
                print("===== Correct ====")
                dis2proto_a = self.calc_distance(size_ltrain, self.trg_train_dl)
                dis2proto_a_test = self.calc_distance(size_ltest, self.trg_test_dl)

                for epoch in range(1, self.hparams["num_epochs"] + 1):
                    joint_loaders = zip(self.src_train_dl, self.trg_train_dl)
                    algorithm.train()

                    for (src_x, src_y, _), (trg_x, _, trg_index) in joint_loaders:
                        src_x, src_y, trg_x, trg_index = src_x.float().to(self.device), src_y.long().to(self.device), \
                                                trg_x.float().to(self.device), trg_index.to(self.device)
                            
                        algorithm.correct(src_x, src_y, trg_x)
                        acc, f1, H = self.evaluate_RAINCOAT(self.trg_test_dl.dataset.y_data)

                dis2proto_c = self.calc_distance(size_ltrain, self.trg_train_dl)
                dis2proto_c_test = self.calc_distance(size_ltest, self.trg_test_dl)
                c_list = self.learn_t(dis2proto_a, dis2proto_c)
                print(c_list)

                self.trg_true_labels = tar_uni_label_test
                acc, f1, H = self.detect_private(dis2proto_a_test, dis2proto_c_test, tar_uni_label_test, c_list)

                log = {'scenario':i,'run_id':run_id,'accuracy':acc,'f1':f1,'H-score':H}
                new_row = pd.DataFrame([log])
                df_c = pd.concat([df_c, new_row], ignore_index=True)
                

        mean_acc, std_acc, mean_f1, std_f1 ,mean_H, std_H = self.avg_result(df_c)
        log = {
            "scenario": "Avg Accuracy: " + str(mean_acc),
            "run_id": "Accuracy STD: " + str(std_acc),
            "accuracy": "Avg F1: " + str(mean_f1),
            "f1": "F1 STD: " + str(std_f1),
            "H-score": "H-score " + str(mean_H) + " H-STD " + str(std_H),
        }

        new_row = pd.DataFrame([log])
        df_c = pd.concat([df_c, new_row], ignore_index=True)
        print(df_c)
        #path = os.path.join(self.exp_log_dir, "average_correct.csv")
        #df_c.to_csv(path, sep=",", index=False)

    def preprocess_labels(self, source_loader, target_loader):
        trg_y= copy.deepcopy(target_loader.dataset.y_data)
        src_y = source_loader.dataset.y_data
        pri_c = np.setdiff1d(trg_y, src_y)
        mask = np.isin(trg_y, pri_c)
        trg_y[mask] = -1
        return trg_y, pri_c
    
    def create_save_dir(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)



    def detect_private(self, d1, d2, tar_uni_label, c_list):
        diff = np.abs(d2-d1)
        for i in range(6):
            cat = np.where(self.trg_pred_labels==i)
            cc = diff[cat]
            if cc.shape[0]>3:
                dip, pval = diptest.diptest(diff[cat])
                if dip < 0.05:
                    print("contain private")
                    # gm = GaussianMixture(n_components=2, random_state=0,max_iter=5000, n_init=50).fit(diff[cat].reshape(-1, 1))
                    # c =  max(gm.means_)
                    # kmeans = KMeans(n_clusters=2, random_state=0,max_iter=5000, n_init=50, init="random").fit(diff[cat].reshape(-1, 1))
                    # c = max(kmeans.cluster_centers_)
                    c = c_list[i]
                    m1 = np.where(diff>c)
                    m2 = np.where(self.trg_pred_labels==i)
                    mask = np.intersect1d(m1, m2)
                    # print(m1, m2, mask)
                    self.trg_pred_labels[mask] = -1
        accuracy = accuracy_score(tar_uni_label, self.trg_pred_labels)
        f1 = f1_score(self.trg_pred_labels, tar_uni_label, pos_label=None, average="macro")
        return accuracy*100, f1, self.H_score()


    def learn_t(self,d1,d2):
        diff = np.abs(d2-d1)
        c_list= []
        for i in range(6):
            cat = np.where(self.trg_train_dl.dataset.y_data==i)
            cc = diff[cat]
            if cc.shape[0]>3:
                dip, pval = diptest.diptest(diff[cat])
                #print(i, dip)
                if dip < 0.05:
                    kmeans = KMeans(n_clusters=2, random_state=0,max_iter=5000, n_init=50, init="random").fit(diff[cat].reshape(-1, 1))
                    c = max(kmeans.cluster_centers_)
                else:
                    c = 1e10
            else: 
                c = 1e10
            c_list.append(c)
        return c_list

    def calc_distance(self, len_y, dataloader):
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)
        feature_extractor.eval()
        classifier.eval()
        
        proto = classifier.logits.weight.data
        # norm = proto.norm(p=2, dim=1, keepdim=True)
        # proto = proto.div(norm.expand_as(norm))
        #trg_drift = np.zeros(len(dataloader.dataset))
        trg_drift = np.zeros(len_y)
        cos = torch.nn.CosineSimilarity(dim=1,eps=1e-6)

        with torch.no_grad():
            for data, labels, trg_index in dataloader:
                data = data.float().to('cuda')
                labels = labels.view((-1)).long().to('cuda')

                features,_ = feature_extractor(data)
                predictions = classifier(features.detach())
                pred_label = torch.argmax(predictions, dim=1)
                proto_M = torch.vstack([proto[l,:] for l in pred_label])
                angle_c = cos(features,proto_M)**2
                # dist = (torch.max(predictions,1).values).div(torch.log(angle_c))
                trg_drift[trg_index] = angle_c.cpu().numpy()
        return trg_drift

    def evaluate_RAINCOAT(self, labels):
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)
        feature_extractor.eval()
        classifier.eval()

        data = copy.deepcopy(self.trg_test_dl.dataset.x_data).float().to('cuda')
        labels = labels.view((-1)).long().to(self.device)

        features, _ = feature_extractor(data)
        predictions = classifier(features)
        
        pred = predictions.argmax(dim=1)
        pred = pred.cpu().numpy()
        accuracy = accuracy_score(labels.cpu().numpy(), pred)
        f1 = f1_score(pred, labels.cpu().numpy(), pos_label=None, average="macro")

        self.trg_pred_labels = pred
        self.trg_true_labels = labels.cpu().numpy()

        return accuracy*100, f1, self.H_score()

    def H_score(self):
        class_c = np.where(self.trg_true_labels!=-1)
        class_p = np.where(self.trg_true_labels==-1)

        
        label_c, pred_c = self.trg_true_labels[class_c], self.trg_pred_labels[class_c]
        label_p, pred_p = self.trg_true_labels[class_p], self.trg_pred_labels[class_p]

        acc_c = accuracy_score(label_c, pred_c)
        acc_p = accuracy_score(label_p, pred_p)
        #print("accuracy of both: ",acc_c, acc_p)
        if acc_c ==0 or acc_p==0:
            #print("acc_C or acc_p is 0")
            H = 0
        else:
            H = 2*acc_c * acc_p/(acc_p+acc_c)
        return H

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

    def get_configs(self):
        dataset_class = get_dataset_class(self.dataset)
        hparams_class = get_hparams_class(self.dataset)
        return dataset_class(), hparams_class()

    def load_data(self, src_id, trg_id):
        self.src_train_dl, self.src_test_dl = data_generator(self.data_path, src_id, self.dataset_configs,
                                                             self.hparams)
        self.trg_train_dl, self.trg_test_dl = data_generator(self.data_path, trg_id, self.dataset_configs,
                                                             self.hparams)
