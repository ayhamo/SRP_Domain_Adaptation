import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix, average_precision_score
from .models_rd import *
from .utils_rd import *
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(1)


def one_hot(y_):
    y_ = y_.reshape(len(y_))

    y_ = [int(x) for x in y_]
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]


def generate_global_structure(data, K=10):
    observations = data[:, :, :36]
    cos_sim = torch.zeros([observations.shape[0], 36, 36])
    for row in tqdm(range(observations.shape[0])):
        unit = observations[row].T
        cos_sim_unit = cosine_similarity(unit)
        cos_sim[row] = torch.from_numpy(cos_sim_unit)

    ave_sim = torch.mean(cos_sim, dim=0)
    index = torch.argsort(ave_sim, dim=0)
    index_K = index < K
    global_structure = index_K * ave_sim
    global_structure = masked_softmax(global_structure)
    return global_structure


def diffuse(unit, N=10):
    n_time = unit.shape[-1]
    keep = n_time//N -1
    unit = unit[:, :keep*N].reshape([-1, keep, N])
    return torch.max(unit, dim=-1).values


def raindrop_training(args):

    # This was added to collect all embeddings and labels for each split
    all_embeddings = []
    all_labels = [] 

    arch = 'raindrop'
    model_path = './Raindrop/models/'

    dataset = args["dataset"]
    print('Dataset used: ', dataset)

    if dataset == 'P12':
        base_path = './Raindrop/P12data'
    elif dataset == 'P19':
        base_path = './Raindrop/P19data'
    elif dataset == 'eICU':
        base_path = './Raindrop/eICUdata'
    elif dataset == 'PAM':
        base_path = './Raindrop/PAMdata'

    baseline = False  # always False for Raindrop
    split = args["splittype"]  # possible values: 'random', 'age', 'gender'
    reverse = args["reverse"]  # False or True
    feature_removal_level = args["feature_removal_level"]  # 'set', 'sample'

    print('args.dataset, args.splittype, args.reverse, args.withmissingratio, args.feature_removal_level',
        args["dataset"], args["splittype"], args["reverse"], args["withmissingratio"], args["feature_removal_level"])

    """While missing_ratio >0, feature_removal_level is automatically used"""
    if args["withmissingratio"] == True:
        missing_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
    else:
        missing_ratios = [0]
    print('missing ratio list', missing_ratios)

    sensor_wise_mask = False

    for missing_ratio in missing_ratios:
        num_epochs = 2
        learning_rate = 0.0001  # 0.001 works slightly better, sometimes 0.0001 better, depends on settings and datasets

        if dataset == 'P12':
            d_static = 9
            d_inp = 36
            static_info = 1
        elif dataset == 'P19':
            d_static = 6
            d_inp = 34
            static_info = 1
        elif dataset == 'eICU':
            d_static = 399
            d_inp = 14
            static_info = 1
        elif dataset == 'PAM':
            d_static = 0
            d_inp = 17
            static_info = None

        d_ob = 4
        d_model = d_inp * d_ob
        nhid = 2 * d_model
        nlayers = 2
        nhead = 2
        dropout = 0.2

        if dataset == 'P12':
            max_len = 215
            n_classes = 2
        elif dataset == 'P19':
            max_len = 60
            n_classes = 2
        elif dataset == 'eICU':
            max_len = 300
            n_classes = 2
        elif dataset == 'PAM':
            max_len = 600
            n_classes = 8

        aggreg = 'mean'

        MAX = 100

        n_runs = 1
        n_splits = 5
        subset = False

        for k in range(n_splits):
            split_idx = k + 1
            print('Split id: %d' % split_idx)
            if dataset == 'P12':
                if subset == True:
                    split_path = '/splits/phy12_split_subset' + str(split_idx) + '.npy'
                else:
                    split_path = '/splits/phy12_split' + str(split_idx) + '.npy'
            elif dataset == 'P19':
                split_path = '/splits/phy19_split' + str(split_idx) + '_new.npy'
            elif dataset == 'eICU':
                split_path = '/splits/eICU_split' + str(split_idx) + '.npy'
            elif dataset == 'PAM':
                split_path = '/splits/PAM_split_' + str(split_idx) + '.npy'

            # prepare the data:
            Ptrain, Pval, Ptest, ytrain, yval, ytest = get_data_split(base_path, split_path, split_type=split, reverse=reverse,
                                                                    baseline=baseline, dataset=dataset,
                                                                    predictive_label=args["predictive_label"],gender=args["gender"])
            print(len(Ptrain), len(Pval), len(Ptest), len(ytrain), len(yval), len(ytest))

            #TODO This line was added to limit the dataset for testing
            Ptrain = Ptrain[:256] 
            ytrain = ytrain[:256]
            Pval = Pval[:256]
            yval = yval[:256]
            Ptest = Ptest[:256]
            ytest = ytest[:256]
            
            if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                T, F = Ptrain[0]['arr'].shape
                D = len(Ptrain[0]['extended_static'])

                Ptrain_tensor = np.zeros((len(Ptrain), T, F))
                Ptrain_static_tensor = np.zeros((len(Ptrain), D))

                for i in range(len(Ptrain)):
                    Ptrain_tensor[i] = Ptrain[i]['arr']
                    Ptrain_static_tensor[i] = Ptrain[i]['extended_static']

                mf, stdf = getStats(Ptrain_tensor)
                ms, ss = getStats_static(Ptrain_static_tensor, dataset=dataset)

                Ptrain_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor = tensorize_normalize(Ptrain, ytrain, mf,
                                                                                                            stdf, ms, ss)
                Pval_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor = tensorize_normalize(Pval, yval, mf, stdf, ms, ss)
                Ptest_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor = tensorize_normalize(Ptest, ytest, mf, stdf, ms,
                                                                                                ss)
            elif dataset == 'PAM':
                T, F = Ptrain[0].shape
                D = 1

                Ptrain_tensor = Ptrain
                Ptrain_static_tensor = np.zeros((len(Ptrain), D))

                mf, stdf = getStats(Ptrain)
                Ptrain_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor = tensorize_normalize_other(Ptrain, ytrain, mf, stdf)
                Pval_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor = tensorize_normalize_other(Pval, yval, mf, stdf)
                Ptest_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor = tensorize_normalize_other(Ptest, ytest, mf, stdf)

            global_structure = torch.ones(d_inp, d_inp)
            
            # Check the shape of the tensors
            #print("Training tensor shape:", Ptrain_tensor.shape)  # Expecting something like (256, T, F) where T is time steps and F is features
            #print("Validation tensor shape:", Pval_tensor.shape)
            #print("Test tensor shape:", Ptest_tensor.shape)

            # remove part of variables in validation and test set
            if missing_ratio > 0:
                num_all_features =int(Pval_tensor.shape[2] / 2)
                num_missing_features = round(missing_ratio * num_all_features)
                if feature_removal_level == 'sample':
                    for i, patient in enumerate(Pval_tensor):
                        idx = np.random.choice(num_all_features, num_missing_features, replace=False)
                        patient[:, idx] = torch.zeros(Pval_tensor.shape[1], num_missing_features)
                        Pval_tensor[i] = patient
                    for i, patient in enumerate(Ptest_tensor):
                        idx = np.random.choice(num_all_features, num_missing_features, replace=False)
                        patient[:, idx] = torch.zeros(Ptest_tensor.shape[1], num_missing_features)
                        Ptest_tensor[i] = patient
                elif feature_removal_level == 'set':
                    density_score_indices = np.load('Raindrop/code/baselines/saved/IG_density_scores_' + dataset + '.npy', allow_pickle=True)[:, 0]
                    idx = density_score_indices[:num_missing_features].astype(int)
                    Pval_tensor[:, :, idx] = torch.zeros(Pval_tensor.shape[0], Pval_tensor.shape[1], num_missing_features)
                    Ptest_tensor[:, :, idx] = torch.zeros(Ptest_tensor.shape[0], Ptest_tensor.shape[1], num_missing_features)

            Ptrain_tensor = Ptrain_tensor.permute(1, 0, 2)
            Pval_tensor = Pval_tensor.permute(1, 0, 2)
            Ptest_tensor = Ptest_tensor.permute(1, 0, 2)

            Ptrain_time_tensor = Ptrain_time_tensor.squeeze(2).permute(1, 0)
            Pval_time_tensor = Pval_time_tensor.squeeze(2).permute(1, 0)
            Ptest_time_tensor = Ptest_time_tensor.squeeze(2).permute(1, 0)

            
            print('- - Run %d - -' % (1))

            if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                model = Raindrop_v2(d_inp, d_model, nhead, nhid, nlayers, dropout, max_len,
                                    d_static, MAX, 0.5, aggreg, n_classes, global_structure,
                                    sensor_wise_mask=sensor_wise_mask)
            elif dataset == 'PAM':
                model = Raindrop_v2(d_inp, d_model, nhead, nhid, nlayers, dropout, max_len,
                                    d_static, MAX, 0.5, aggreg, n_classes, global_structure,
                                    sensor_wise_mask=sensor_wise_mask, static=False)

            model = model.cuda()

            criterion = torch.nn.CrossEntropyLoss().cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1,
                                                                patience=1, threshold=0.0001, threshold_mode='rel',
                                                                cooldown=0, min_lr=1e-8, eps=1e-08, verbose=True)

            idx_0 = np.where(ytrain == 0)[0]
            idx_1 = np.where(ytrain == 1)[0]

            if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                strategy = 2
            elif dataset == 'PAM':
                strategy = 3

            n0, n1 = len(idx_0), len(idx_1)
            expanded_idx_1 = np.concatenate([idx_1, idx_1, idx_1], axis=0)
            expanded_n1 = len(expanded_idx_1)

            batch_size = 128
            if strategy == 1:
                n_batches = 10
            elif strategy == 2:
                K0 = n0 // int(batch_size / 2)
                K1 = expanded_n1 // int(batch_size / 2)
                n_batches = np.min([K0, K1])
            elif strategy == 3:
                n_batches = 30

            best_auc_val = 0.0
            print('Stop epochs: %d, Batches/epoch: %d, Total batches: %d' % (num_epochs, n_batches, num_epochs * n_batches))

            best_model_state_dict = None  # Variable to store the best model's state dictionary

            for epoch in range(num_epochs):
                model.train()

                if strategy == 2:
                    np.random.shuffle(expanded_idx_1)
                    I1 = expanded_idx_1
                    np.random.shuffle(idx_0)
                    I0 = idx_0

                for n in range(n_batches):
                    if strategy == 1:
                        idx = random_sample(idx_0, idx_1, batch_size)
                    elif strategy == 2:
                        idx0_batch = I0[n * int(batch_size / 2):(n + 1) * int(batch_size / 2)]
                        idx1_batch = I1[n * int(batch_size / 2):(n + 1) * int(batch_size / 2)]
                        idx = np.concatenate([idx0_batch, idx1_batch], axis=0)
                    elif strategy == 3:
                        idx = np.random.choice(list(range(Ptrain_tensor.shape[1])), size=int(batch_size), replace=False)
                        # idx = random_sample_8(ytrain, batch_size)   # to balance dataset

                    if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                        P, Ptime, Pstatic, y = Ptrain_tensor[:, idx, :].cuda(), Ptrain_time_tensor[:, idx].cuda(), \
                                            Ptrain_static_tensor[idx].cuda(), ytrain_tensor[idx].cuda()
                    elif dataset == 'PAM':
                        P, Ptime, Pstatic, y = Ptrain_tensor[:, idx, :].cuda(), Ptrain_time_tensor[:, idx].cuda(), \
                                            None, ytrain_tensor[idx].cuda()

                    lengths = torch.sum(Ptime > 0, dim=0)

                    outputs, local_structure_regularization, _ = model.forward(P, Pstatic, Ptime, lengths)

                    optimizer.zero_grad()
                    loss = criterion(outputs, y)
                    loss.backward()
                    optimizer.step()

                if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                    train_probs = torch.squeeze(torch.sigmoid(outputs))
                    train_probs = train_probs.cpu().detach().numpy()
                    train_y = y.cpu().detach().numpy()

                elif dataset == 'PAM':
                    train_probs = torch.squeeze(nn.functional.softmax(outputs, dim=1))
                    train_probs = train_probs.cpu().detach().numpy()
                    train_y = y.cpu().detach().numpy()

                if epoch == num_epochs - 1:
                    print("Training appending")
                    all_embeddings.append(model.get_sensor_embeddings())
                    all_labels.append(ytrain_tensor)
                    #print(confusion_matrix(train_y, np.argmax(train_probs, axis=1), labels=[0, 1]))

                """Validation"""
                model.eval()
                if epoch == 0 or epoch % 1 == 0:
                    with torch.no_grad():
                        out_val = evaluate_standard(model, Pval_tensor, Pval_time_tensor, Pval_static_tensor, static=static_info)
                        out_val = torch.squeeze(torch.sigmoid(out_val))
                        out_val = out_val.detach().cpu().numpy()

                        val_loss = criterion(torch.from_numpy(out_val), torch.from_numpy(yval.squeeze(1)).long())

                        if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                            auc_val = roc_auc_score(yval, out_val[:, 1])
                            aupr_val = average_precision_score(yval, out_val[:, 1])
                        elif dataset == 'PAM':
                            auc_val = roc_auc_score(one_hot(yval), out_val)
                            aupr_val = average_precision_score(one_hot(yval), out_val)

                        print("Validation: Epoch %d,  val_loss:%.4f, aupr_val: %.2f, auc_val: %.2f" % (epoch,
                                                                                                        val_loss.item(),
                                                                                                        aupr_val * 100,
                                                                                                        auc_val * 100))

                        scheduler.step(aupr_val)
                        if auc_val > best_auc_val:
                            best_auc_val = auc_val
                            print(
                                "**[S] Epoch %d, aupr_val: %.4f, auc_val: %.4f **" % (
                                epoch, aupr_val * 100, auc_val * 100))
                            best_model_state_dict = model.state_dict()  # Store the best model's state dict

                        if epoch == num_epochs - 1:
                            print("Validation appending")
                            all_embeddings.append(model.get_sensor_embeddings())
                            all_labels.append(yval_tensor)
            
            # to save model once
            torch.save(best_model_state_dict, model_path + arch + '_' + str(split_idx) + '.pt')

            """testing"""
            # No need for testing really here

        # # save in numpy file
        # np.save('./results/' + arch + '_phy12_setfunction.npy', [acc_vec, auprc_vec, auroc_vec])

        return {"samples": all_embeddings, "labels": all_labels}
