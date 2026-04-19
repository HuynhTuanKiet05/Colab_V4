import timeit
import argparse
import numpy as np
import pandas as pd
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as fn
import os
import gc

# Import from the root modules
from data_preprocess_improved import get_data, data_processing, k_fold, dgl_similarity_graph, dgl_heterograph
from model.improved.improved_model import AMNTDDA
from metric import get_metric

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EarlyStopping:
    def __init__(self, patience=200, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_auc):
        score = val_auc
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k_fold', type=int, default=10, help='k-fold cross validation')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight_decay')
    parser.add_argument('--random_seed', type=int, default=1234, help='random seed')
    parser.add_argument('--neighbor', type=int, default=5, help='neighbor (paper optimal k=5)')
    parser.add_argument('--negative_rate', type=float, default=1.0, help='negative_rate')
    parser.add_argument('--dataset', default='C-dataset', help='dataset')
    parser.add_argument('--dropout', default=0.2, type=float, help='dropout')
    
    # Model dimensions
    parser.add_argument('--hgt_in_dim', default=128, type=int, help='HGT input dimension (optimized for 4GB GPU)')
    parser.add_argument('--hgt_layer', default=3, type=int, help='HGT layers')
    parser.add_argument('--hgt_head', default=8, type=int, help='HGT heads')
    parser.add_argument('--gt_layer', default=2, type=int, help='GT layers')
    parser.add_argument('--gt_head', default=2, type=int, help='GT heads')
    parser.add_argument('--gt_out_dim', default=128, type=int, help='GT output dimension (optimized for 4GB GPU)')
    parser.add_argument('--tr_layer', default=2, type=int, help='Transformer layers')
    parser.add_argument('--tr_head', default=4, type=int, help='Transformer heads')
    
    parser.add_argument('--patience', default=200, type=int, help='early stopping patience')

    args = parser.parse_args()
    
    # Setup directories
    args.data_dir = 'AMDGT_original/data/' + args.dataset + '/'
    args.result_dir = 'Result/improved/' + args.dataset + '/'
    os.makedirs(args.result_dir, exist_ok=True)

    print(f"--- Starting Final Improved Pipeline ---")
    print(f"Dataset: {args.dataset} | LR: {args.lr} | Dim: {args.gt_out_dim} | Neighbor: {args.neighbor}")

    # Data loading
    data = get_data(args)
    args.drug_number = data['drug_number']
    args.disease_number = data['disease_number']
    args.protein_number = data['protein_number']

    data = data_processing(data, args)
    data = k_fold(data, args)

    # Similarity Graphs
    drdr_graph, didi_graph, data = dgl_similarity_graph(data, args)
    drdr_graph = drdr_graph.to(device)
    didi_graph = didi_graph.to(device)

    # Base Features
    drug_feature = torch.FloatTensor(data['drugfeature']).to(device)
    disease_feature = torch.FloatTensor(data['diseasefeature']).to(device)
    protein_feature = torch.FloatTensor(data['proteinfeature']).to(device)

    Metric_Header = ('Epoch\t\tTime\t\tAUC\t\tAUPR\t\tAccuracy\t\tPrecision\t\tRecall\t\tF1-score\t\tMcc')
    AUCs, AUPRs, Accs, Precs, Recs, F1s, MCCs, Epochs = [], [], [], [], [], [], [], []

    for i in range(args.k_fold):
        print(f'\n--- Fold: {i} ---')
        print(Metric_Header)

        # Initialize Improved Model
        model = AMNTDDA(args).to(device)
        optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        early_stopping = EarlyStopping(patience=args.patience, verbose=False)

        best_auc = 0
        best_metrics = None
        
        X_train = torch.LongTensor(data['X_train'][i]).to(device)
        Y_train = torch.LongTensor(data['Y_train'][i]).to(device)
        X_test = torch.LongTensor(data['X_test'][i]).to(device)
        Y_test = data['Y_test'][i].flatten()

        # Weighting
        n_pos = torch.sum(Y_train).item()
        n_neg = Y_train.numel() - n_pos
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, n_neg/n_pos]).to(device))

        # Heterograph with 6 etypes
        drdipr_graph, data = dgl_heterograph(data, data['X_train'][i], args)
        drdipr_graph = drdipr_graph.to(device)
        
        # Cleanup memory after building large graph
        gc.collect()
        torch.cuda.empty_cache()

        start = timeit.default_timer()

        for epoch in range(args.epochs):
            model.train()
            # Forward pass
            _, train_score = model(drdr_graph, didi_graph, drdipr_graph, drug_feature, disease_feature, protein_feature, X_train)
            train_loss = criterion(train_score, torch.flatten(Y_train))
            
            optimizer.zero_grad()
            train_loss.backward()
            
            # Gradient Clipping to prevent CUBLAS error 13
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            scheduler.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                model.eval()
                with torch.no_grad():
                    _, test_score = model(drdr_graph, didi_graph, drdipr_graph, drug_feature, disease_feature, protein_feature, X_test)
                
                test_prob = fn.softmax(test_score, dim=-1)[:, 1].cpu().numpy()
                test_pred = torch.argmax(test_score, dim=-1).cpu().numpy()

                AUC, AUPR, accuracy, precision, recall, f1, mcc = get_metric(Y_test, test_pred, test_prob)
                
                if AUC > best_auc:
                    best_auc = AUC
                    best_metrics = (AUC, AUPR, accuracy, precision, recall, f1, mcc, epoch + 1)
                    torch.save(model.state_dict(), os.path.join(args.result_dir, f'best_model_fold_{i}.pth'))

                early_stopping(AUC)
                
                time_now = timeit.default_timer() - start
                print(f'{epoch+1}\t\t{time_now:.2f}\t\t{AUC:.5f}\t\t{AUPR:.5f}\t\t{accuracy:.5f}\t\t{precision:.5f}\t\t{recall:.5f}\t\t{f1:.5f}\t\t{mcc:.5f}')

                if early_stopping.early_stop:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        if best_metrics:
            AUCs.append(best_metrics[0]); AUPRs.append(best_metrics[1]); Accs.append(best_metrics[2])
            Precs.append(best_metrics[3]); Recs.append(best_metrics[4]); F1s.append(best_metrics[5])
            MCCs.append(best_metrics[6]); Epochs.append(best_metrics[7])
            print(f'Fold {i} Best AUC: {best_metrics[0]:.5f} (Epoch {best_metrics[7]})')
        
        # Proper cleanup
        del model, optimizer, scheduler, drdipr_graph
        gc.collect()
        torch.cuda.empty_cache()

    # Final Results Processing
    results_df = pd.DataFrame({
        'Fold': [f'Fold {i}' for i in range(len(AUCs))],
        'Best_Epoch': Epochs,
        'AUC': AUCs, 'AUPR': AUPRs, 'Accuracy': Accs, 
        'Precision': Precs, 'Recall': Recs, 'F1-score': F1s, 'Mcc': MCCs
    })
    
    metrics_only = results_df.drop(columns=['Fold', 'Best_Epoch'])
    summary_df = pd.DataFrame([['Mean', ''] + metrics_only.mean().tolist(), ['Std', ''] + metrics_only.std().tolist()], columns=results_df.columns)
    final_df = pd.concat([results_df, summary_df], ignore_index=True)
    
    print('\n' + '='*30 + '\nFINAL RESULTS SUMMARY (IMPROVED PIPELINE)\n' + '='*30)
    print(final_df.iloc[-2:])
    
    csv_path = os.path.join(args.result_dir, '10_fold_results_improved.csv')
    final_df.to_csv(csv_path, index=False)
    print(f'\nKết quả cải tiến đã được lưu tại: {csv_path}')
