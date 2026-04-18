import timeit
import argparse
import os
import warnings
import numpy as np
import pandas as pd
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as fn
from data_preprocess import *
from model.AMNTDDA import AMNTDDA
from metric import *

device = torch.device(os.environ.get('AMDGT_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))

if device.type != 'cuda':
    warnings.filterwarnings(
        'ignore',
        message="User provided device_type of 'cuda', but CUDA is not available. Disabling",
        module='torch.autocast_mode',
    )


def should_evaluate(epoch, total_epochs, eval_every):
    return epoch == 1 or epoch == total_epochs or epoch % eval_every == 0


def monitored_metric_value(metric_name, auc_value, aupr_value):
    if metric_name == 'auc':
        return auc_value
    if metric_name == 'aupr':
        return aupr_value
    raise ValueError(f'Unsupported metric for early stopping: {metric_name}')


def evaluate_model(model, drdr_graph, didi_graph, drdipr_graph, drug_feature, disease_feature, protein_feature, x_eval, y_eval):
    with torch.no_grad():
        model.eval()
        _, eval_score = model(
            drdr_graph,
            didi_graph,
            drdipr_graph,
            drug_feature,
            disease_feature,
            protein_feature,
            x_eval,
        )

    eval_prob = fn.softmax(eval_score, dim=-1)
    eval_pred = torch.argmax(eval_score, dim=-1)

    eval_prob = eval_prob[:, 1].cpu().numpy()
    eval_pred = eval_pred.cpu().numpy()

    return get_metric(y_eval, eval_pred, eval_prob)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--k_fold', type=int, default=10, help='k-fold cross validation')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight_decay')
    parser.add_argument('--random_seed', type=int, default=1234, help='random seed')
    parser.add_argument('--neighbor', type=int, default=20, help='neighbor')
    parser.add_argument('--negative_rate', type=float, default=1.0, help='negative_rate')
    parser.add_argument('--dataset', default='C-dataset', help='dataset')
    parser.add_argument('--dropout', default='0.2', type=float, help='dropout')
    parser.add_argument('--gt_layer', default='2', type=int, help='graph transformer layer')
    parser.add_argument('--gt_head', default='2', type=int, help='graph transformer head')
    parser.add_argument('--gt_out_dim', default='200', type=int, help='graph transformer output dimension')
    parser.add_argument('--hgt_layer', default='2', type=int, help='heterogeneous graph transformer layer')
    parser.add_argument('--hgt_head', default='8', type=int, help='heterogeneous graph transformer head')
    parser.add_argument('--hgt_in_dim', default='64', type=int, help='heterogeneous graph transformer input dimension')
    parser.add_argument('--hgt_head_dim', default='25', type=int, help='heterogeneous graph transformer head dimension')
    parser.add_argument('--hgt_out_dim', default='200', type=int, help='heterogeneous graph transformer output dimension')
    parser.add_argument('--tr_layer', default='2', type=int, help='transformer layer')
    parser.add_argument('--tr_head', default='4', type=int, help='transformer head')
    parser.add_argument('--eval_every', type=int, default=10, help='evaluate every N epochs instead of every epoch')
    parser.add_argument('--early_stop_start_epoch', type=int, default=400, help='start checking early stopping after this epoch')
    parser.add_argument('--early_stop_patience', type=int, default=100, help='stop if monitored metric does not improve for this many epochs after the start epoch')
    parser.add_argument('--early_stop_min_delta', type=float, default=1e-4, help='minimum metric improvement to reset early stopping')
    parser.add_argument('--early_stop_metric', choices=['auc', 'aupr'], default='aupr', help='metric used to track best epoch and early stopping')
    parser.add_argument('--disable_early_stop', action='store_true', help='disable early stopping and always run all epochs')
    parser.add_argument('--use_relation_attention', action='store_true', default=True, help='use relation-aware attention in HGT')
    parser.add_argument('--use_metapath', action='store_true', default=True, help='use explicit metapath branch in HGT')
    parser.add_argument('--use_global_hgt', action='store_true', default=True, help='use global context branch in HGT')
    parser.add_argument('--use_topological', action='store_true', default=True, help='use topological branch in HGT')
    parser.add_argument('--disable_relation_attention', action='store_false', dest='use_relation_attention', help='disable relation-aware attention in HGT')
    parser.add_argument('--disable_metapath', action='store_false', dest='use_metapath', help='disable explicit metapath branch in HGT')
    parser.add_argument('--disable_global_hgt', action='store_false', dest='use_global_hgt', help='disable global context branch in HGT')
    parser.add_argument('--disable_topological', action='store_false', dest='use_topological', help='disable topological branch in HGT')

    args = parser.parse_args()
    args.eval_every = max(1, args.eval_every)
    args.data_dir = 'data/' + args.dataset + '/'
    args.result_dir = 'Result/' + args.dataset + '/AMNTDDA/'

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    data = get_data(args)
    args.drug_number = data['drug_number']
    args.disease_number = data['disease_number']
    args.protein_number = data['protein_number']

    data = data_processing(data, args)
    data = k_fold(data, args)

    drdr_graph, didi_graph, data = dgl_similarity_graph(data, args)

    drdr_graph = drdr_graph.to(device)
    didi_graph = didi_graph.to(device)

    drug_feature = torch.FloatTensor(data['drugfeature']).to(device)
    disease_feature = torch.FloatTensor(data['diseasefeature']).to(device)
    protein_feature = torch.FloatTensor(data['proteinfeature']).to(device)
    all_sample = torch.tensor(data['all_drdi']).long()

    total_start = timeit.default_timer()

    cross_entropy = nn.CrossEntropyLoss()

    Metric = ('Epoch\t\tTime\t\tAUC\t\tAUPR\t\tAccuracy\t\tPrecision\t\tRecall\t\tF1-score\t\tMcc')
    AUCs, AUPRs = [], []

    print('Dataset:', args.dataset)
    print('Device:', device)
    print('Eval every:', args.eval_every, '| Early stop metric:', args.early_stop_metric)
    if args.disable_early_stop:
        print('Early stopping: disabled')
    else:
        print(
            'Early stopping: start at epoch',
            args.early_stop_start_epoch,
            '| patience:',
            args.early_stop_patience,
            '| min_delta:',
            args.early_stop_min_delta,
        )

    for i in range(args.k_fold):

        print('fold:', i)
        print(Metric)

        model = AMNTDDA(args)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

        best_epoch = 0
        best_auc, best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc = 0, 0, 0, 0, 0, 0, 0
        best_monitor_metric = float('-inf')
        last_improve_epoch = 0
        X_train = torch.LongTensor(data['X_train'][i]).to(device)
        Y_train = torch.LongTensor(data['Y_train'][i]).to(device)
        X_test = torch.LongTensor(data['X_test'][i]).to(device)
        Y_test = data['Y_test'][i].flatten()

        drdipr_graph, data = dgl_heterograph(data, data['X_train'][i], args)
        drdipr_graph = drdipr_graph.to(device)

        fold_start = timeit.default_timer()
        epochs_run = 0
        for epoch in range(args.epochs):
            current_epoch = epoch + 1
            model.train()
            _, train_score = model(drdr_graph, didi_graph, drdipr_graph, drug_feature, disease_feature, protein_feature, X_train)
            train_loss = cross_entropy(train_score, torch.flatten(Y_train))
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            epochs_run = current_epoch

            if not should_evaluate(current_epoch, args.epochs, args.eval_every):
                continue

            AUC, AUPR, accuracy, precision, recall, f1, mcc = evaluate_model(
                model,
                drdr_graph,
                didi_graph,
                drdipr_graph,
                drug_feature,
                disease_feature,
                protein_feature,
                X_test,
                Y_test,
            )

            end = timeit.default_timer()
            time = end - fold_start
            show = [current_epoch, round(time, 2), round(AUC, 5), round(AUPR, 5), round(accuracy, 5),
                       round(precision, 5), round(recall, 5), round(f1, 5), round(mcc, 5)]
            print('\t\t'.join(map(str, show)))
            current_monitor_metric = monitored_metric_value(args.early_stop_metric, AUC, AUPR)
            if current_monitor_metric > best_monitor_metric + args.early_stop_min_delta:
                best_epoch = current_epoch
                best_monitor_metric = current_monitor_metric
                best_auc = AUC
                best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc = AUPR, accuracy, precision, recall, f1, mcc
                last_improve_epoch = current_epoch
                print(
                    args.early_stop_metric.upper(),
                    'improved at epoch',
                    best_epoch,
                    ';\tbest_auc:',
                    round(best_auc, 5),
                    ';\tbest_aupr:',
                    round(best_aupr, 5),
                )

            if not args.disable_early_stop and current_epoch >= args.early_stop_start_epoch:
                if last_improve_epoch < args.early_stop_start_epoch:
                    last_improve_epoch = args.early_stop_start_epoch
                if current_epoch - last_improve_epoch >= args.early_stop_patience:
                    print(
                        'Early stopping triggered at epoch',
                        current_epoch,
                        'because',
                        args.early_stop_metric.upper(),
                        'did not improve for',
                        args.early_stop_patience,
                        'epochs after epoch',
                        args.early_stop_start_epoch,
                    )
                    break

        AUCs.append(best_auc)
        AUPRs.append(best_aupr)
        print(
            'Best fold metrics:',
            best_auc,
            best_aupr,
            best_accuracy,
            best_precision,
            best_recall,
            best_f1,
            best_mcc,
            '| best_epoch:',
            best_epoch,
            '| epochs_run:',
            epochs_run,
        )

    print('AUC:', AUCs)
    AUC_mean = np.mean(AUCs)
    AUC_std = np.std(AUCs)
    print('Mean AUC:', AUC_mean, '(', AUC_std, ')')

    print('AUPR:', AUPRs)
    AUPR_mean = np.mean(AUPRs)
    AUPR_std = np.std(AUPRs)
    print('Mean AUPR:', AUPR_mean, '(', AUPR_std, ')')
    print('Total training time (s):', round(timeit.default_timer() - total_start, 2))



