import argparse
import os
import datetime
import csv
import time
import pickle
import json
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from models.SwinTST import SwinTST
from models.ResNet import ResNet
from datasets import MRIDataset, generate_and_save_splits, get_dataloaders_from_splits, GroupShuffleSplit, Subset, DataLoader, get_dynamic_transforms


def calculate_metrics(y_true, y_pred, labels=None):
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    specificity = recall_score(y_true, y_pred, pos_label=0)
    sensitivity = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, average='weighted')

    return balanced_acc, recall, precision, specificity, sensitivity, f1


def evaluate_model(model, loader, criterion, device):
    """Helper function to evaluate model performance on a data loader."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            # Ensure labels are 1D LongTensor
            if len(labels.shape) > 1:
                if labels.shape[1] > 1:
                    labels = torch.argmax(labels, dim=1)
                elif labels.shape[1] == 1:
                    labels = labels.squeeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
    metrics = calculate_metrics(all_labels, all_preds)
    accuracy = 100. * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels) if len(all_labels) > 0 else 0.0

    return avg_loss, accuracy, metrics, all_labels, all_preds


def train_model(model, train_loader, val_loader, test_loader, output_path, num_epochs=200,
                checkpoint_dir='checkpoints', logs_dir="logs", class_weights=None, checkpoint_suffix="", lr=1e-4,
                checkpoint_saving=True, seed=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))

    optimizer = Adam(model.parameters(), lr=lr)

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    if not os.path.exists(os.path.join(output_path, checkpoint_dir)):
        os.makedirs(os.path.join(output_path, checkpoint_dir))
    if not os.path.exists(os.path.join(output_path, logs_dir)):
        os.makedirs(os.path.join(output_path, logs_dir))
    if not os.path.exists(os.path.join(output_path, logs_dir, "CM")):
        os.makedirs(os.path.join(output_path, logs_dir, "CM"))



    currDate = datetime.datetime.now().strftime("%m-%d-%H-%M")
    log_file = os.path.join(output_path, logs_dir, f'metrics_log{checkpoint_suffix}_{currDate}.csv')
    with open(log_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Train Balanced Acc', 'Train Recall',
                            'Train Precision', 'Train Specificity', 'Train Sensitivity', 'Train F1',
                            'Val Loss', 'Val Accuracy', 'Val Balanced Acc', 'Val Recall',
                            'Val Precision', 'Val Specificity', 'Val Sensitivity', 'Val F1',
                            'Test Loss', 'Test Accuracy', 'Test Balanced Acc', 'Test Recall',
                            'Test Precision', 'Test Specificity', 'Test Sensitivity', 'Test F1'])

    # --- Training Loop ---
    best_val_bacc = -1.0
    best_epoch = -1
    best_val_results = {'loss': 0, 'acc': 0, 'metrics': (0.0,) * 6}
    best_test_results = {'loss': 0, 'acc': 0, 'metrics': (0.0,) * 6}
    best_epoch_cm = {'train': None, 'val': None, 'test': None}

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        all_train_preds = []
        all_train_labels = []

        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            if len(labels.shape) > 1 and labels.shape[1] > 1:
                labels = torch.argmax(labels, dim=1)
            if len(labels.shape) == 1:
                labels.unsqueeze(-1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
            loss.backward()
            optimizer.step()

        scheduler.step()  # Step the scheduler once per epoch

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100. * train_correct / train_total
        train_metrics = calculate_metrics(all_train_labels, all_train_preds)
        train_cm = train_metrics[-1]

        avg_val_loss, val_accuracy, val_metrics, val_labels, val_preds = evaluate_model(model, val_loader, criterion, device)
        val_cm = val_metrics[-1]

        # Evaluate on test set (informational during fold training)
        if test_loader:
             avg_test_loss, test_accuracy, test_metrics, _, _ = evaluate_model(model, test_loader, criterion, device)
             test_cm = test_metrics[-1]
             test_metrics = test_metrics[:-1]
        else:
            avg_test_loss, test_accuracy, test_metrics = 0.0, 0.0, (0.0,) * 6
            test_cm = None

        print(f'Epoch {epoch + 1}, '
              f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, '
              f'Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')
        print(
            f'Train_BACC: {train_metrics[0]:.2f}, Val_BACC: {val_metrics[0]:.2f}, Test_BACC: {test_metrics[0]:.2f}')

        # Log metrics to CSV file
        with open(log_file, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([epoch + 1, avg_train_loss, train_accuracy] + list(train_metrics) +
                               [avg_val_loss, val_accuracy] + list(val_metrics) +
                               [avg_test_loss, test_accuracy] + list(test_metrics))

        # --- Checkpoint Saving ---
        current_val_bacc = val_metrics[0]
        if checkpoint_saving and (current_val_bacc > best_val_bacc and epoch > 30 or (current_val_bacc >= best_val_bacc and epoch > 30 and test_accuracy > best_test_results['acc'])):
            best_val_bacc = current_val_bacc
            best_epoch = epoch + 1
            best_val_results = {'loss': avg_val_loss, 'acc': val_accuracy, 'metrics': val_metrics}
            best_test_results = {'loss': avg_test_loss, 'acc': test_accuracy, 'metrics': test_metrics}
            best_epoch_cm['train'] = train_cm
            best_epoch_cm['val'] = val_cm
            best_epoch_cm['test'] = test_cm
            # Save the model checkpoint
            checkpoint_path = os.path.join(output_path, checkpoint_dir,
                                           f'statedict{checkpoint_suffix}_epoch-{epoch}_best.pth')

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_bacc': val_metrics[0],
                'test_bacc': test_metrics[0],
                'val_indices': val_loader.dataset.indices,
                'best_epoch_cm': best_epoch_cm,
                'best_val_results': best_val_results,
                'best_test_results': best_test_results
            }
            if test_loader is not None:
                checkpoint['test_indices'] = test_loader.dataset.indices
            if final_model:
                checkpoint["y_true_final"] = np.array(val_labels)
                checkpoint["y_pred_final"] = np.array(val_preds)
                
            torch.save(checkpoint, checkpoint_path)

    print(f"Finished training for {checkpoint_suffix}. Best Val BACC: {best_val_bacc:.4f} at epoch {best_epoch}")
    return best_val_results, best_test_results, best_epoch



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train dMRI AD Classifier with K-Fold CV")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset directory containing phenotypes and input pickle")
    parser.add_argument("--output_path", type=str, required=True, help="Base path for all outputs (logs, checkpoints, splits)")
    parser.add_argument("--input_details", type=str, nargs=2, required=True,
                        help="Input pickle filename (relative to data_path) and input type (e.g., 'noddi', 'DTI')")
    parser.add_argument("--verbose", action="store_true", help="Print verbose loading/debugging information")
    parser.add_argument("--gpu_id", type=int, default=1, help="Choose specific GPU")
    parser.add_argument("--target_variable", type=str, nargs="+", required=True,
                        help="Target variable(s) (e.g., 'AD', 'MCI', 'MCI AD' or 'CN' for amyloid)")
    parser.add_argument("--control_variable", type=str, nargs="+", required=True,
                        help="Control variable(s) (e.g., 'CN')")
    parser.add_argument("--amyloid", type=str, default="C2T2", choices=["C0T1", "C1T0", "C2T2"],
                        help="Amyloid classification mode. C0T1: A+ vs A- (1 vs 0), C1T0: A- vs A+ (1 vs 0), C2T2: Standard group classification (ignore amyloid).")
    parser.add_argument("--model", type=str, choices=['resnet18', 'resnet34', 'swint'], required=True,
                        help="Model architecture")
    parser.add_argument("--model_freeze", type=str, required=True, help="Model Freezing strategy (e.g., 'allButStage4', 'all', 'None', 'LoRA+r4+a1+d0.1+attn')")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-1, help="Learning rate")
    parser.add_argument("--dynamic_data_augmentation", type=str, default="F",
                        help="Dynamic data augmentation (e.g., 'F' or 'T+t444+r0+n0.05')")
    parser.add_argument("--seed", type=int, default=42, help="Seed for all random operations (splits, weights, etc.)")
    parser.add_argument("--kfold", type=int, default=5, help="Number of folds for cross-validation (set to 1 for train/val/test only)")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs for training")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Fraction of subjects for the fixed test set (set to 0 to disable test set)")
    parser.add_argument("--force_regen_dataset", action="store_true",
                        help="Force regeneration of the pickled dataset file")
    parser.add_argument("--force_regen_splits", action="store_true", help="Force regeneration of the split file")
    args = parser.parse_args()

    assert args.amyloid in ["C0T1", "C1T0", "C2T2"]

    split_dir = os.path.join(args.output_path, "splits")
    log_cv_dir = os.path.join(args.output_path, "logs_CV")
    log_folds_dir = os.path.join(args.output_path, "logs")
    log_final_dir = os.path.join(args.output_path, "logs_final")
    checkpoint_dir = os.path.join(args.output_path, "checkpoints")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    target_variables_str = ",".join(args.target_variable) if len(args.target_variable) > 1 else args.target_variable[0]
    control_variables_str = ",".join(args.control_variable) if len(args.control_variable) > 1 else \
        args.control_variable[0]

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)

    MRIDataset_file = (f"input-{args.input_details[1]}_target-{args.target_variable}_control-{args.control_variable}"
                       f"_cuda-cpu_amyloid-{args.amyloid}_MRIDataset.pickle")
    MRIDataset_path = os.path.join(args.data_path, MRIDataset_file)

    if not os.path.exists(MRIDataset_path):
        print("Generating dataset...")
        start_time = time.time()
        dataset = MRIDataset(data_path=args.data_path,
                             input_details=args.input_details,
                             verbose=True,
                             target_variable=args.target_variable,
                             control_variable=args.control_variable,
                             device='cpu',
                             amyloid=args.amyloid)
        elapsed_time = time.time() - start_time
        print(elapsed_time)
        with open(MRIDataset_path, "wb") as f:
            pickle.dump(dataset, f)
        import sys

        sys.exit()
    else:
        start_time = time.time()
        print("Starting loading pickled dataset")
        with open(MRIDataset_path, "rb") as f:
            dataset = pickle.load(f)
        elapsed_time = time.time() - start_time
        print(elapsed_time)

    dataset.inputs = dataset.inputs.to("cuda")
    dataset.outputs = dataset.outputs.to("cuda")
    if dataset.class_weights is not None:
        dataset.class_weights = dataset.class_weights.to("cuda")

    base_checkpoint_suffix = (f"_{args.model}_input-{args.input_details[1]}_target-{target_variables_str}_"
                              f"control-{control_variables_str}_bs-{args.batch_size}_lr-{args.learning_rate}_"
                              f"freeze-{args.model_freeze}_dyn-{args.dynamic_data_augmentation}_amyloid-{args.amyloid}_seed-{args.seed}")

    # --- Generate or Load Splits ---
    split_filename = f"splits_target-{target_variables_str}_control-{control_variables_str}_k{args.kfold}_test{args.test_ratio:.2f}_seed-{args.seed}.json"
    if args.force_regen_splits and os.path.exists(os.path.join(split_dir, split_filename)):
        print(f"Force regenerating splits, deleting existing file: {split_filename}")
        try:
            os.remove(os.path.join(split_dir, split_filename))
        except OSError as e:
            print(f"Error deleting existing split file: {e}")

    splits_data = generate_and_save_splits(dataset=dataset,
                                           kfold=args.kfold,
                                           test_ratio=args.test_ratio,
                                           seed=args.seed,
                                           output_split_dir=split_dir,
                                           split_filename=split_filename)
    if splits_data is None:
        print("Error: Failed to generate or load data splits. Aborting.")
        sys.exit(1)

    if dataset.class_weights is not None:
        num_classes = len(dataset.class_weights)
    else:
        num_classes = 1

    # --- K-Fold Cross-Validation ---
    fold_val_results_list = []  # Store BACC from best epoch of each fold's validation set
    fold_test_results_list = []  # Store BACC from test set corresponding to best epoch of each fold's validation set


    for fold_idx in range(args.kfold):
        print(f"\n===== Starting Fold {fold_idx + 1}/{args.kfold} =====")

        # Get DataLoaders for the current fold
        train_loader, val_loader, test_loader, train_weights = get_dataloaders_from_splits(
                dataset,
                batch_size=args.batch_size,
                fold_index=fold_idx,
                splits_data=splits_data,
                dynamic_data_augmentation=args.dynamic_data_augmentation
            )

        # Initialize model for each fold
        if args.model in ['swint']:
            model = SwinTST(num_classes=num_classes, size=args.model, freeze_mode=args.model_freeze)
        elif args.model in ['resnet18', 'resnet34']:
            model = ResNet(num_classes=num_classes, size=args.model, freeze_mode=args.model_freeze)
        else:
            raise Exception("unknown model")

        fold_checkpoint_suffix = f"_fold-{fold_idx + 1}of{args.kfold}{base_checkpoint_suffix}"

        # Train the model for the current fold
        val_metrics_results, test_metrics_results, _  = train_model(model, train_loader, val_loader, test_loader, args.output_path,
                    num_epochs=args.num_epochs,  class_weights=train_weights.tolist(),
                    checkpoint_suffix=fold_checkpoint_suffix,
                    lr=args.learning_rate, checkpoint_saving=False, seed=args.seed)

        fold_val_results_list.append(val_metrics_results['metrics'])  # Store tuple of metrics
        fold_test_results_list.append(test_metrics_results['metrics'])  # Store tuple of metrics

    # --- CV Summary ---
    print("\n===== Cross-Validation Summary =====")
    val_metrics_array = np.array(fold_val_results_list)
    test_metrics_array = np.array(fold_test_results_list)

    avg_val_metrics = np.mean(val_metrics_array, axis=0)
    std_val_metrics = np.std(val_metrics_array, axis=0)
    avg_test_metrics = np.mean(test_metrics_array, axis=0)
    std_test_metrics = np.std(test_metrics_array, axis=0)

    metric_names = ['BACC', 'Recall', 'Precision', 'Specificity', 'Sensitivity', 'F1']
    print("Average Validation Metrics across folds:")
    for name, avg, std in zip(metric_names, avg_val_metrics, std_val_metrics):
        print(f"  {name:<12}: {avg:.4f} +/- {std:.4f}")

    print("\nAverage Test Metrics (from best val epoch) across folds:")
    for name, avg, std in zip(metric_names, avg_test_metrics, std_test_metrics):
        print(f"  {name:<12}: {avg:.4f} +/- {std:.4f}")

    # --- Log CV Summary ---
    summary_log_file = os.path.join(log_cv_dir, f'cross_val_summary{base_checkpoint_suffix}.csv')
    if not os.path.exists(log_cv_dir):
        os.makedirs(log_cv_dir)
    with open(summary_log_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        header = ['Fold'] + [f'Val {m}' for m in metric_names] + [f'Test {m}' for m in metric_names]
        csvwriter.writerow(header)
        for i in range(args.kfold):
            row = [i + 1] + list(fold_val_results_list[i]) + list(fold_test_results_list[i])
            csvwriter.writerow(row)
        csvwriter.writerow([]) # Empty line separator
        avg_row = ['Average'] + list(avg_val_metrics) + list(avg_test_metrics)
        std_row = ['Std Dev'] + list(std_val_metrics) + list(std_test_metrics)
        csvwriter.writerow(avg_row)
        csvwriter.writerow(std_row)
    print(f"Cross-validation summary saved to: {summary_log_file}")

    # --- Final Training and Evaluation on Test Set ---
    if args.test_ratio > 0:
        print(f"\n===== Starting Final Training on combined Train+Val set =====")

        # 1. Get Combined Train/Val and Test Indices from splits_data
        # Combine all 'train' and 'val' indices from the folds
        final_train_indices = []
        for fold_split in splits_data['folds']:
            # Make sure keys exist before extending
            if 'train' in fold_split: final_train_indices.extend(fold_split['train'])
            if 'val' in fold_split: final_train_indices.extend(fold_split['val'])
        # Ensure uniqueness (although SGKF should prevent overlap between folds)
        final_train_indices = sorted(list(set(final_train_indices)))

        final_test_indices = splits_data['test_indices']

        # Get subject counts for info
        final_train_subjects = set()
        for fold_subj_split in splits_data.get('fold_subjects', []):
            if 'train' in fold_subj_split: final_train_subjects.update(fold_subj_split['train'])
            if 'val' in fold_subj_split: final_train_subjects.update(fold_subj_split['val'])
        final_test_subjects = splits_data.get('test_subjects', [])

        print(
            f"Final Training set size: {len(final_train_indices)} samples ({len(final_train_subjects)} unique subjects)")
        print(f"Final Test set size: {len(final_test_indices)} samples ({len(final_test_subjects)} unique subjects)")



        if len(final_train_indices) == 0 or len(final_test_indices) == 0:
            raise ValueError("Resulting train or test set is empty after final split.")

        # 2. Create DataLoaders
        # Apply dynamic augmentation only to the final training set
        original_transform = dataset.transform
        if args.dynamic_data_augmentation.startswith("T"):
            dynamic_transforms = get_dynamic_transforms(args.dynamic_data_augmentation)
            dataset.transform = dynamic_transforms
        else:
            dataset.transform = None

        final_train_dataset = Subset(dataset, final_train_indices)
        final_train_loader = DataLoader(final_train_dataset, batch_size=args.batch_size, shuffle=True)

        # Reset transform for test set
        dataset.transform = None
        final_test_dataset = Subset(dataset, final_test_indices)
        final_test_loader = DataLoader(final_test_dataset, batch_size=args.batch_size, shuffle=False)
        dataset.transform = original_transform  # Restore original

        # 3. Compute Class Weights for combined training set
        final_train_weights = dataset.compute_class_weights(final_train_indices)
        print(f"Final Train Class weights: {final_train_weights}")

        # 4. Initialize Final Model
        if args.model == 'swint':
            final_model = SwinTST(num_classes=num_classes, size=args.model, freeze_mode=args.model_freeze)
        elif args.model in ['resnet18', 'resnet34']:
            final_model = ResNet(num_classes=num_classes, size=args.model, freeze_mode=args.model_freeze)
        else:  # Should not happen due to choices constraint
            raise Exception("unknown model")

        final_checkpoint_suffix = f"_final{base_checkpoint_suffix}"

        # 5. Train Final Model
        print("Training final model")
        _, final_test_results_during_train, best_final_epoch = train_model(
            final_model, final_train_loader, final_test_loader, None, args.output_path,
            num_epochs=args.num_epochs,
            class_weights=final_train_weights,
            logs_dir="logs_final",
            checkpoint_suffix=final_checkpoint_suffix,
            lr=args.learning_rate,
            checkpoint_saving=True,
            seed=args.seed
        )

        print(f"Final model trained. Best epoch: {best_final_epoch}")


    print("\n===== Script Finished =====")
