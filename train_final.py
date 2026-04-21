import argparse
import gc
import logging
import os
import random
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_preprocess_improved import dgl_heterograph, dgl_similarity_graph, data_processing, get_data, k_fold
from metric import get_metric
from model.improved.improved_model import AMNTDDA
from topology_features import extract_topology_features


REQUIRED_DATA_FILES = [
    "DrugFingerprint.csv",
    "DrugGIP.csv",
    "DiseasePS.csv",
    "DiseaseGIP.csv",
    "DrugDiseaseAssociationNumber.csv",
    "DrugProteinAssociationNumber.csv",
    "ProteinDiseaseAssociationNumber.csv",
    "Drug_mol2vec.csv",
    "DiseaseFeature.csv",
    "Protein_ESM.csv",
]

DATASET_PRESETS = {
    "B-dataset": {
        "lr": 1e-4,
        "weight_decay": 1e-3,
        "neighbor": 3,
        "gt_out_dim": 512,
        "hgt_layer": 2,
        "hgt_in_dim": 512,
        "hgt_head_dim": 64,
        "topo_hidden": 192,
    },
    "C-dataset": {
        "lr": 1e-4,
        "weight_decay": 1e-3,
        "neighbor": 5,
        "gt_out_dim": 256,
        "hgt_layer": 2,
        "hgt_in_dim": 256,
        "hgt_head_dim": 32,
        "topo_hidden": 128,
    },
    "F-dataset": {
        "lr": 1e-4,
        "weight_decay": 1e-3,
        "neighbor": 5,
        "gt_out_dim": 256,
        "hgt_layer": 2,
        "hgt_in_dim": 256,
        "hgt_head_dim": 32,
        "topo_hidden": 128,
    },
}


def resolve_device(device_name):
    if device_name == "auto":
        device_name = os.environ.get("AMDGT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        device_name = "cpu"
    return torch.device(device_name)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        import dgl

        dgl.seed(seed)
    except Exception:
        pass


def validate_data_dir(data_dir):
    missing = [name for name in REQUIRED_DATA_FILES if not os.path.exists(os.path.join(data_dir, name))]
    if missing:
        raise FileNotFoundError(f"Missing dataset files in {data_dir}: {', '.join(missing)}")


def apply_dataset_preset(args):
    preset = DATASET_PRESETS.get(args.dataset, {})
    for key, value in preset.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    if args.hgt_head_dim is None:
        args.hgt_head_dim = max(1, args.gt_out_dim // args.hgt_head)
    args.hgt_out_dim = args.gt_out_dim
    args.topo_feat_dim = 7


def build_model_tag(args):
    return f"{args.assoc_backbone}_{args.fusion_mode}_{args.pair_mode}_{args.gate_mode}"


def save_checkpoint(path, model, optimizer, scheduler, fold_idx, epoch, best_metrics, args):
    checkpoint = {
        "fold": fold_idx,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metrics": best_metrics,
        "args": vars(args),
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(checkpoint, path)


def configure_logging(result_dir):
    log_file = Path(result_dir) / "training_log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return log_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_fold", type=int, default=10, help="k-fold cross validation")
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs to train")
    parser.add_argument("--lr", type=float, default=None, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=None, help="weight decay")
    parser.add_argument("--random_seed", type=int, default=1234, help="random seed")
    parser.add_argument("--neighbor", type=int, default=None, help="k for similarity knn graphs")
    parser.add_argument("--negative_rate", type=float, default=1.0, help="negative sampling rate")
    parser.add_argument("--dataset", default="C-dataset", help="dataset")
    parser.add_argument("--dropout", default=0.2, type=float, help="dropout")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="training device")
    parser.add_argument("--data_root", default=None, help="dataset directory; defaults to AMDGT_original/data/<dataset>")
    parser.add_argument("--result_root", default=None, help="output directory; defaults to Result/improved/<dataset>")
    parser.add_argument("--save_checkpoints", action=argparse.BooleanOptionalAction, default=False, help="save model checkpoints")
    parser.add_argument("--eval_start_epoch", default=1, type=int, help="minimum epoch before evaluation begins")
    parser.add_argument("--score_every", default=1, type=int, help="evaluate every N epochs after eval_start_epoch")
    parser.add_argument("--patience", default=0, type=int, help="early stopping patience in evaluation steps; <=0 disables early stopping")
    parser.add_argument("--lambda_cl", default=0.1, type=float, help="weight for contrastive alignment loss")
    parser.add_argument("--temperature", default=0.5, type=float, help="temperature for contrastive loss")
    parser.add_argument("--disable_scheduler", action="store_true", help="disable ReduceLROnPlateau scheduler")
    parser.add_argument("--topo_hidden", default=None, type=int, help="hidden dimension for topology encoder")
    parser.add_argument("--fold_limit", type=int, default=None, help="optional limit on number of folds to execute")
    parser.add_argument("--assoc_backbone", choices=["vanilla_hgt", "rlghgt"], default="vanilla_hgt", help="association encoder backbone")
    parser.add_argument("--fusion_mode", choices=["mva", "rvg", "mva_fuzzy"], default="mva", help="node-view fusion strategy")
    parser.add_argument("--pair_mode", choices=["mul_mlp", "interaction"], default="mul_mlp", help="pair scoring head")
    parser.add_argument("--gate_mode", choices=["scalar", "vector"], default="vector", help="fuzzy gate output type")
    parser.add_argument("--gate_bias_init", default=-2.0, type=float, help="initial bias for fuzzy gate")
    parser.add_argument("--grad_clip", default=0.0, type=float, help="max gradient norm; <=0 disables clipping")
    parser.add_argument("--use_relation_attention", action=argparse.BooleanOptionalAction, default=True, help="enable relation-aware attention in RLGHGT")
    parser.add_argument("--use_metapath", action=argparse.BooleanOptionalAction, default=True, help="enable metapath branch in RLGHGT")
    parser.add_argument("--use_global_hgt", action=argparse.BooleanOptionalAction, default=True, help="enable global context branch in RLGHGT")
    parser.add_argument("--use_topological", action=argparse.BooleanOptionalAction, default=True, help="enable topology projection branch in RLGHGT")

    parser.add_argument("--hgt_in_dim", default=None, type=int, help="HGT input dimension")
    parser.add_argument("--hgt_layer", default=None, type=int, help="HGT layers")
    parser.add_argument("--hgt_head", default=8, type=int, help="HGT heads")
    parser.add_argument("--hgt_head_dim", default=None, type=int, help="HGT per-head output dimension")
    parser.add_argument("--hgt_out_dim", default=None, type=int, help=argparse.SUPPRESS)
    parser.add_argument("--gt_layer", default=2, type=int, help="GT layers")
    parser.add_argument("--gt_head", default=2, type=int, help="GT heads")
    parser.add_argument("--gt_out_dim", default=None, type=int, help="GT output dimension")
    parser.add_argument("--tr_layer", default=2, type=int, help="Transformer layers")
    parser.add_argument("--tr_head", default=4, type=int, help="Transformer heads")

    parser.add_argument("--min_lr", default=1e-6, type=float, help=argparse.SUPPRESS)
    parser.add_argument("--lr_warmup_epochs", default=40, type=int, help=argparse.SUPPRESS)
    parser.add_argument("--warmup_epochs", default=220, type=int, help=argparse.SUPPRESS)
    parser.add_argument("--log_every", default=25, type=int, help=argparse.SUPPRESS)
    parser.add_argument("--focal_gamma", default=1.2, type=float, help=argparse.SUPPRESS)
    parser.add_argument("--focal_gamma_warm", default=2.0, type=float, help=argparse.SUPPRESS)
    parser.add_argument("--contrastive_weight", default=0.08, type=float, help=argparse.SUPPRESS)
    parser.add_argument("--contrastive_temperature", default=0.20, type=float, help=argparse.SUPPRESS)
    parser.add_argument("--ranking_weight", default=0.08, type=float, help=argparse.SUPPRESS)
    parser.add_argument("--ranking_margin", default=0.20, type=float, help=argparse.SUPPRESS)
    parser.add_argument("--ranking_samples", default=2048, type=int, help=argparse.SUPPRESS)
    parser.add_argument("--hard_negative_weight", default=1.2, type=float, help=argparse.SUPPRESS)
    parser.add_argument("--label_smoothing", default=0.01, type=float, help=argparse.SUPPRESS)
    parser.add_argument("--target_auc", default=0.96, type=float, help=argparse.SUPPRESS)
    parser.add_argument("--target_auc_warmup", default=400, type=int, help=argparse.SUPPRESS)
    parser.add_argument("--target_auc_patience", default=4, type=int, help=argparse.SUPPRESS)
    parser.add_argument("--plateau_patience", default=3, type=int, help=argparse.SUPPRESS)
    parser.add_argument("--plateau_factor", default=0.5, type=float, help=argparse.SUPPRESS)
    parser.add_argument("--ema_decay", default=0.995, type=float, help=argparse.SUPPRESS)

    args = parser.parse_args()
    apply_dataset_preset(args)
    if args.fusion_mode == "mva_fuzzy":
        args.fusion_mode = "rvg"

    device = resolve_device(args.device)
    args.device = device
    os.environ["AMDGT_DEVICE"] = device.type
    set_random_seed(args.random_seed)

    default_data_dir = Path("AMDGT_original") / "data" / args.dataset
    default_result_dir = Path("Result") / "improved" / args.dataset
    args.data_dir = str(Path(args.data_root) if args.data_root else default_data_dir)
    args.result_dir = str(Path(args.result_root) if args.result_root else default_result_dir)
    args.model_tag = build_model_tag(args)

    validate_data_dir(args.data_dir)
    os.makedirs(args.result_dir, exist_ok=True)
    log_file = configure_logging(args.result_dir)

    logging.info("--- Starting Final Improved Pipeline ---")
    logging.info(f"Dataset: {args.dataset} | LR: {args.lr} | GT dim: {args.gt_out_dim} | Neighbor: {args.neighbor}")
    logging.info(f"Device: {device} | Data dir: {args.data_dir} | Result dir: {args.result_dir}")
    logging.info(f"Save checkpoints: {args.save_checkpoints}")
    if args.patience > 0:
        logging.info(f"Early stopping patience: {args.patience}")
    else:
        logging.info("Early stopping: disabled")
    logging.info(f"Contrastive weight lambda_cl: {args.lambda_cl}")
    logging.info(
        "Model config: "
        f"tag={args.model_tag} | assoc={args.assoc_backbone} | fusion={args.fusion_mode} | "
        f"pair={args.pair_mode} | gate={args.gate_mode}"
    )
    logging.info(f"Training log file: {log_file}")

    data = get_data(args)
    args.drug_number = data["drug_number"]
    args.disease_number = data["disease_number"]
    args.protein_number = data["protein_number"]
    logging.info(f"Loaded data: {args.drug_number} drugs, {args.disease_number} diseases, {args.protein_number} proteins")

    data = data_processing(data, args)
    data = k_fold(data, args)

    drdr_graph, didi_graph, data = dgl_similarity_graph(data, args)
    drdr_graph = drdr_graph.to(device)
    didi_graph = didi_graph.to(device)

    logging.info("Extracting topology features...")
    drug_topo_feat, disease_topo_feat = extract_topology_features(data, args)
    drug_topo_feat = drug_topo_feat.to(device)
    disease_topo_feat = disease_topo_feat.to(device)
    logging.info(f"Drug topology features: {tuple(drug_topo_feat.shape)}")
    logging.info(f"Disease topology features: {tuple(disease_topo_feat.shape)}")

    drug_feature = torch.tensor(data["drugfeature"], dtype=torch.float32).to(device)
    disease_feature = torch.tensor(data["diseasefeature"], dtype=torch.float32).to(device)
    protein_feature = torch.tensor(data["proteinfeature"], dtype=torch.float32).to(device)

    metric_header = "Epoch\t\tTime\t\tLR\t\tLoss\t\tCL_Loss\t\tAUC\t\tAUPR\t\tAccuracy\t\tPrecision\t\tRecall\t\tF1-score\t\tMcc"
    logging.info(metric_header)

    aucs, auprs, accs, precs, recs, f1s, mccs, best_epochs = [], [], [], [], [], [], [], []
    best_overall_auc = -1.0
    best_overall_path = None
    best_overall_payload = None

    max_folds = args.k_fold if args.fold_limit is None else min(args.k_fold, args.fold_limit)

    for fold_idx in range(max_folds):
        logging.info(f'\n{"=" * 60}')
        logging.info(f"Fold: {fold_idx}")
        logging.info(f'{"=" * 60}')
        logging.info(metric_header)

        model = AMNTDDA(args).to(device)
        optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
        scheduler = None if args.disable_scheduler else ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=30,
            min_lr=1e-6,
        )
        criterion = nn.CrossEntropyLoss()

        x_train = torch.tensor(data["X_train"][fold_idx], dtype=torch.long, device=device)
        y_train = torch.tensor(data["Y_train"][fold_idx], dtype=torch.long, device=device).flatten()
        x_test = torch.tensor(data["X_test"][fold_idx], dtype=torch.long, device=device)
        y_test = data["Y_test"][fold_idx].flatten()

        drdipr_graph, _, _ = dgl_heterograph(data, data["X_train"][fold_idx], args)
        drdipr_graph = drdipr_graph.to(device)

        best_fold = {
            "AUC": -1.0,
            "AUPR": 0.0,
            "Accuracy": 0.0,
            "Precision": 0.0,
            "Recall": 0.0,
            "F1": 0.0,
            "MCC": 0.0,
            "epoch": 0,
        }
        no_improve_steps = 0
        start = timeit.default_timer()

        for epoch in range(args.epochs):
            model.train()
            _, train_score, aux_losses = model(
                drdr_graph,
                didi_graph,
                drdipr_graph,
                drug_feature,
                disease_feature,
                protein_feature,
                x_train,
                drug_topo_feat=drug_topo_feat,
                disease_topo_feat=disease_topo_feat,
                return_aux=True,
            )
            contrastive_loss = aux_losses.get("contrastive", train_score.new_tensor(0.0))
            ce_loss = criterion(train_score, y_train)
            train_loss = ce_loss + args.lambda_cl * contrastive_loss

            optimizer.zero_grad()
            train_loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            should_eval = (epoch + 1) >= args.eval_start_epoch and ((epoch + 1 - args.eval_start_epoch) % max(1, args.score_every) == 0)
            if should_eval or epoch == args.epochs - 1:
                model.eval()
                with torch.no_grad():
                    _, test_score, diagnostics = model(
                        drdr_graph,
                        didi_graph,
                        drdipr_graph,
                        drug_feature,
                        disease_feature,
                        protein_feature,
                        x_test,
                        drug_topo_feat=drug_topo_feat,
                        disease_topo_feat=disease_topo_feat,
                        return_diagnostics=True,
                    )

                test_prob = fn.softmax(test_score, dim=-1)[:, 1].cpu().numpy()
                test_pred = torch.argmax(test_score, dim=-1).cpu().numpy()
                auc, aupr, accuracy, precision, recall, f1, mcc = get_metric(y_test, test_pred, test_prob)
                if scheduler is not None:
                    scheduler.step(auc)

                elapsed = timeit.default_timer() - start
                current_lr = optimizer.param_groups[0]["lr"]
                logging.info(
                    "\t\t".join(
                        map(
                            str,
                            [
                                epoch + 1,
                                round(elapsed, 2),
                                f"{current_lr:.1e}",
                                round(float(train_loss.item()), 5),
                                round(float(contrastive_loss.item()), 5),
                                round(float(auc), 5),
                                round(float(aupr), 5),
                                round(float(accuracy), 5),
                                round(float(precision), 5),
                                round(float(recall), 5),
                                round(float(f1), 5),
                                round(float(mcc), 5),
                            ],
                        )
                    )
                )

                if auc > best_fold["AUC"] + 1e-6:
                    best_fold.update(
                        {
                            "AUC": float(auc),
                            "AUPR": float(aupr),
                            "Accuracy": float(accuracy),
                            "Precision": float(precision),
                            "Recall": float(recall),
                            "F1": float(f1),
                            "MCC": float(mcc),
                            "epoch": epoch + 1,
                        }
                    )
                    no_improve_steps = 0
                    if args.save_checkpoints:
                        checkpoint_path = os.path.join(args.result_dir, f"best_model_{args.model_tag}_fold{fold_idx}.pth")
                        save_checkpoint(checkpoint_path, model, optimizer, scheduler, fold_idx, epoch + 1, best_fold, args)
                        if auc > best_overall_auc:
                            best_overall_auc = float(auc)
                            best_overall_path = checkpoint_path
                            best_overall_payload = {
                                "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                                "optimizer_state_dict": optimizer.state_dict(),
                                "best_metrics": best_fold.copy(),
                                "fold": fold_idx,
                                "epoch": epoch + 1,
                                "args": vars(args),
                            }
                            if scheduler is not None:
                                best_overall_payload["scheduler_state_dict"] = scheduler.state_dict()
                    else:
                        if auc > best_overall_auc:
                            best_overall_auc = float(auc)
                    logging.info(f"AUC improved at epoch {epoch + 1} ;\tbest_auc: {auc:.5f}")
                else:
                    no_improve_steps += max(1, args.score_every)

                if args.patience > 0 and no_improve_steps >= args.patience:
                    logging.info(f"Early stopping at epoch {epoch + 1} after {no_improve_steps} epochs without AUC improvement.")
                    break
            elif (epoch + 1) % max(1, args.log_every) == 0:
                elapsed = timeit.default_timer() - start
                current_lr = optimizer.param_groups[0]["lr"]
                logging.info(
                    f"Epoch {epoch + 1:4d} | {elapsed:7.2f}s | "
                    f"lr {current_lr:.1e} | loss {train_loss.item():.5f} | cls {ce_loss.item():.5f} | ctr {contrastive_loss.item():.5f}"
                )

        aucs.append(best_fold["AUC"])
        auprs.append(best_fold["AUPR"])
        accs.append(best_fold["Accuracy"])
        precs.append(best_fold["Precision"])
        recs.append(best_fold["Recall"])
        f1s.append(best_fold["F1"])
        mccs.append(best_fold["MCC"])
        best_epochs.append(best_fold["epoch"])
        logging.info(f"Fold {fold_idx} completed: best_epoch={best_fold['epoch']}, best_auc={best_fold['AUC']:.5f}, best_aupr={best_fold['AUPR']:.5f}")

        del model, optimizer, scheduler, drdipr_graph
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(
        {
            "Fold": [f"Fold {i}" for i in range(len(aucs))],
            "Best_Epoch": best_epochs,
            "AUC": aucs,
            "AUPR": auprs,
            "Accuracy": accs,
            "Precision": precs,
            "Recall": recs,
            "F1-score": f1s,
            "Mcc": mccs,
            "ModelTag": [args.model_tag for _ in aucs],
        }
    )
    summary_df = pd.DataFrame(
        {
            "Fold": ["Mean", "Std"],
            "Best_Epoch": ["", ""],
            "AUC": [np.mean(aucs), np.std(aucs)],
            "AUPR": [np.mean(auprs), np.std(auprs)],
            "Accuracy": [np.mean(accs), np.std(accs)],
            "Precision": [np.mean(precs), np.std(precs)],
            "Recall": [np.mean(recs), np.std(recs)],
            "F1-score": [np.mean(f1s), np.std(f1s)],
            "Mcc": [np.mean(mccs), np.std(mccs)],
            "ModelTag": [args.model_tag, args.model_tag],
        }
    )
    final_df = pd.concat([results_df, summary_df], ignore_index=True)

    output_csv = os.path.join(args.result_dir, f"{max_folds}_fold_results_{args.model_tag}.csv")
    final_df.to_csv(output_csv, index=False)
    logging.info(f'\n{"=" * 60}')
    logging.info("FINAL RESULTS SUMMARY (REFERENCE-ALIGNED PIPELINE)")
    logging.info(f'{"=" * 60}')
    logging.info(f"AUC: {aucs}")
    logging.info(f"Mean AUC: {float(np.mean(aucs)):.5f} (+/- {float(np.std(aucs)):.5f})")
    logging.info(f"AUPR: {auprs}")
    logging.info(f"Mean AUPR: {float(np.mean(auprs)):.5f} (+/- {float(np.std(auprs)):.5f})")
    logging.info(f"\n{summary_df.to_string(index=False)}")
    logging.info(f"Saved improved results to: {output_csv}")

    if args.save_checkpoints and best_overall_payload is not None:
        overall_path = os.path.join(args.result_dir, f"best_model_{args.model_tag}.pth")
        torch.save(best_overall_payload, overall_path)
        logging.info(f"Saved best overall checkpoint to: {overall_path}")
