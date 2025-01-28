import argparse
import os
import datetime
import csv
import time
import pickle
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, f1_score
from models.SwinTST import SwinTST
from models.ResNet import ResNet
from datasets import MRIDataset, get_train_val_test_dataloaders


def calculate_metrics(y_true, y_pred):
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    specificity = recall_score(y_true, y_pred, pos_label=0, average='weighted')
    sensitivity = recall_score(y_true, y_pred, pos_label=1, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return balanced_acc, recall, precision, specificity, sensitivity, f1


def train_model(model, train_loader, val_loader, test_loader, output_path, num_epochs=200,
                checkpoint_dir='checkpoints', logs_dir="logs", class_weights=None, checkpoint_suffix="", lr=1e-4):

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

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100. * train_correct / train_total
        train_metrics = calculate_metrics(all_train_labels, all_train_preds)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                if len(labels.shape) > 1 and labels.shape[1] > 1:
                    labels = torch.argmax(labels, dim=1)
                if len(labels.shape) == 1:
                    labels.unsqueeze(-1)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())


        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100. * val_correct / val_total
        val_metrics = calculate_metrics(all_val_labels, all_val_preds)


        scheduler.step()

        # Test
        test_loss = 0
        test_correct = 0
        test_total = 0
        all_test_preds = []
        all_test_labels = []

        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                if len(labels.shape) > 1 and labels.shape[1] > 1:
                    labels = torch.argmax(labels, dim=1)
                if len(labels.shape) == 1:
                    labels.unsqueeze(-1)
                outputs = model(inputs)
                test_loss += criterion(outputs, labels).item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
                all_test_preds.extend(predicted.cpu().numpy())
                all_test_labels.extend(labels.cpu().numpy())

        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = 100. * test_correct / test_total
        test_metrics = calculate_metrics(all_test_labels, all_test_preds)

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

        # Save the model checkpoint
        checkpoint_path = os.path.join(output_path, checkpoint_dir,
                                       f'statedict{checkpoint_suffix}_epoch-{epoch}_best.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'val_bacc': val_metrics[0],
            'test_bacc': test_metrics[0],
            'val_indices': val_loader.dataset.indices,
            'test_indices': test_loader.dataset.indices
        }, checkpoint_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train dMRI AD Classifier")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output dir")
    parser.add_argument("--input_details", type=str, nargs=2, required=True,
                        help="List of input details (must contain exactly two items)")
    parser.add_argument("--verbose", action="store_true", help="Print debug information")
    parser.add_argument("--gpu_id", type=int, default=1, help="Choose specific GPU")
    parser.add_argument("--target_variable", type=str, nargs="+", required=True, default=['MCI', 'AD'],
                        help="Target variables")
    parser.add_argument("--control_variable", type=str, nargs="+", required=True, default=['CN'],
                        help="Control variables")
    parser.add_argument("--model", type=str, choices=['resnet18', 'resnet34', 'swint'], required=True,
                        help="Model architecture")
    parser.add_argument("--model_freeze", type=str, required=True, help="Model Freezing")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-1, help="Learning rate")
    parser.add_argument("--dynamic_data_augmentation", type=str, default="F",
                        help="Dynamic data augmentation details")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the split (train,val,test)")
    parser.add_argument("--amyloid", type=str, default="C2T2", help="Perform amyloid classification if not default")
    args = parser.parse_args()

    assert args.amyloid in ["C0T1", "C1T0", "C2T2"]

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

    train_loader, val_loader, test_loader = get_train_val_test_dataloaders(dataset, batch_size=args.batch_size,
                                                                           val_ratio=0.15, test_ratio=0.15,
                                                                           seed=args.seed,
                                                                           dynamic_data_augmentation=args.dynamic_data_augmentation)

    if args.model in ['swint']:
        if dataset.class_weights is not None:
            num_classes = len(dataset.class_weights)
        else:
            num_classes = 1
        model = SwinTST(num_classes=num_classes, size=args.model, freeze_mode=args.model_freeze)
    elif args.model in ['resnet18', 'resnet34']:
        if dataset.class_weights is not None:
            num_classes = len(dataset.class_weights)
        else:
            num_classes = 1
        model = ResNet(num_classes=num_classes, size=args.model, freeze_mode=args.model_freeze)
    else:
        raise Exception("unknown model")

    checkpoint_suffix = (f"_{args.model}_input-{args.input_details[1]}_target-{target_variables_str}_"
                         f"control-{control_variables_str}_batch-{args.batch_size}_lr-{args.learning_rate}_"
                         f"balance-{args.balance_method}_freeze-{args.model_freeze}_"
                         f"dynamic-{args.dynamic_data_augmentation}_amyloid-{args.amyloid}")

    train_model(model, train_loader, val_loader, test_loader, args.output_path,
                class_weights=dataset.class_weights,
                checkpoint_suffix=checkpoint_suffix,
                lr=args.learning_rate)
