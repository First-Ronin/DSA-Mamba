import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm

from model.DSAmamba import VSSM as dsamamba  # import model


import rl_plotter
from rl_plotter import logger
import argparse
import medmnist
from medmnist import INFO, Evaluator
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--num-classes', default=6, type=int)
    parser.add_argument('--n-channels', default=3, type=int)
    # parser.add_argument('--model-name', default='dsamamba', type=str)
    parser.add_argument('--num-works', default=4, type=int)
    parser.add_argument('--env-name', default='FETAL', type=str)
    parser.add_argument('--train-dataset-path', default='kvasir-dataset-v2/train',
                        type=str)
    parser.add_argument('--val-dataset-path', default='kvasir-dataset-v2/val', type=str)
    parser.add_argument('--medmnist', default=False, type=bool)
    parser.add_argument('--medmnist-download', default=False, type=bool)
    parser.add_argument('--medmnist-choice', default='octmnist', type=str,
                        choices=['pathmnist', 'chestmnist', 'dermamnist', 'octmnist',
                                 'pneumoniamnist', 'retinamnist', 'breastmnist', 'organmnist_axial',
                                 'organmnist_coronal', 'organmnist_sagittal', 'organamnist', 'organcmnist',
                                 'organsmnist'])

    return parser


def main():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    model_name = 'dsamamba'
    print(f'Current model: {model_name}')
    print(f'Current datasets: {args.env_name}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    if args.medmnist is False:
        data_transform = {
            "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
            "val": transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
        # TODO
        train_dataset = datasets.ImageFolder(root=args.train_dataset_path,
                                             transform=data_transform["train"])
        train_num = len(train_dataset)

        flower_list = train_dataset.class_to_idx
        cla_dict = dict((val, key) for key, val in flower_list.items())
        # write dict into json file
        json_str = json.dumps(cla_dict, indent=4)
        with open('class_indices.json', 'w') as json_file:
            json_file.write(json_str)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_works)

        validate_dataset = datasets.ImageFolder(root=args.val_dataset_path,
                                                transform=data_transform["val"])
        val_num = len(validate_dataset)
        validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                      batch_size=args.batch_size, shuffle=False,
                                                      num_workers=args.num_works)
        print("using {} images for training, {} images for validation.".format(train_num, val_num))
    else:
        print('use medmnist datasets')
        info = INFO[args.medmnist_choice]
        # task = info['task']
        args.n_channels = info['n_channels']
        args.num_classes = len(info['label'])
        # print('len labels {}'.format(len(info['label'])))
        DataClass = getattr(medmnist, info['python_class'])

        data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

        # Load datasets
        custom_root = 'kvasir-dataset-v2'

        train_dataset = DataClass(split='train', transform=data_transform, download=args.medmnist_download,
                                  root=custom_root)
        train_num = len(train_dataset)
        val_dataset = DataClass(split='val', transform=data_transform, download=args.medmnist_download,
                                root=custom_root)
        val_num = len(val_dataset)
        # test_dataset = DataClass(split='test', transform=data_transform, download=args.medmnist_download)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_works)
        validate_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                                      num_workers=args.num_works)
        # test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_works)
        print("using {} images for training, {} images for validation.".format(train_num, val_num))

    print(f"Instantiating {model_name} model...")
    net = dsamamba(in_chans=args.n_channels, num_classes=args.num_classes)

    best_acc = 0.0
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)  # 0.0001

    acc_logger = rl_plotter.logger.Logger(exp_name=model_name, env_name=args.env_name + '_acc')
    auc_logger = rl_plotter.logger.Logger(exp_name=model_name, env_name=args.env_name + '_auc')
    pre_logger = rl_plotter.logger.Logger(exp_name=model_name, env_name=args.env_name + '_precision')
    sen_logger = rl_plotter.logger.Logger(exp_name=model_name, env_name=args.env_name + '_sensitivity')
    f1s_logger = rl_plotter.logger.Logger(exp_name=model_name, env_name=args.env_name + '_f1score')
    spe_logger = rl_plotter.logger.Logger(exp_name=model_name, env_name=args.env_name + '_specificity')


    save_path = f'./pth_out/{model_name}_FETAL_best.pth'
    os.makedirs('./pth_out', exist_ok=True)

    train_steps = len(train_loader)
    for epoch in range(args.epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for steps, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()

            labels = labels.squeeze().long().to(device)
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     args.epochs,
                                                                     loss)

        # validate
        val_accurate, auc, precision, sensitivity, f1, specificity = calculate_metrics(net, validate_loader, val_num,
                                                                                       device=device)

        print(
            '[epoch %d] train_loss: %.3f  val_accuracy: %.3f, auc: %.3f, precision: %.3f, sensitivity: %.3f, f1: %.3f, specificity: %.3f' %
            (epoch + 1, running_loss / train_steps, val_accurate, auc, precision, sensitivity, f1, specificity))

        acc_logger.update(score=[val_accurate], total_steps=epoch + 1)
        auc_logger.update(score=[auc], total_steps=epoch + 1)
        pre_logger.update(score=[precision], total_steps=epoch + 1)
        sen_logger.update(score=[sensitivity], total_steps=epoch + 1)
        f1s_logger.update(score=[f1], total_steps=epoch + 1)
        spe_logger.update(score=[specificity], total_steps=epoch + 1)

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
            print(f"New best model saved to {save_path} with accuracy: {best_acc:.4f}")

    print('Finished Training')


# Function to calculate metrics
def calculate_metrics(model, validate_loader, val_num, device):
    model.eval()
    acc = 0.0  # accumulate accurate number / epoch
    pred_scores = []
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        val_bar = tqdm(validate_loader, file=sys.stdout, desc="Validating")
        for val_data in val_bar:
            val_images, val_labels = val_data
            val_labels = val_labels.squeeze().to(device)

            outputs = model(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]

            acc += torch.eq(predict_y, val_labels).sum().item()
            probabilities = torch.softmax(outputs, dim=1)

            all_labels.extend(val_labels.cpu().numpy())
            all_predictions.extend(predict_y.cpu().numpy())
            pred_scores.extend(probabilities.cpu().numpy())

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    pred_scores = np.array(pred_scores)

    val_accurate = acc / val_num

    # Check if it's a multi-class or binary problem for AUC
    if pred_scores.shape[1] > 2:
        auc = roc_auc_score(all_labels, pred_scores, multi_class='ovr', average='macro')
    else:  # Binary case
        auc = roc_auc_score(all_labels, pred_scores[:, 1])

    # Calculate Precision, Recall (Sensitivity), and F1-score
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)  # Sensitivity
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)

    cm = confusion_matrix(all_labels, all_predictions)
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)


    specificity_per_class = tn / (tn + fp)
    specificity_per_class[np.isnan(specificity_per_class)] = 0
    specificity = np.mean(specificity_per_class)

    return val_accurate, auc, precision, recall, f1, specificity


if __name__ == '__main__':
    main()