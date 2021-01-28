import argparse
import csv
import glob
import os
import random
from collections import defaultdict
from torchvision.models import *

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from retina_dataset_simple import Retina_Dataset_Simple
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--data_dir', type=str, default="data")
    parser.add_argument('--ckpt_dir', type=str, default="ckpt")
    parser.add_argument('--name', type=str, default="baseline")

    # Model
    parser.add_argument('--model', type=str, default="resnet50")
    parser.add_argument('--seed', type=int, default=random.randint(0, 9999999))

    # Training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch_per', type=int, default=99)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--task_binary', type=bool, default=True)
    parser.add_argument('--early_stop', type=int, default=10)

    # Incremental
    parser.add_argument('--num_exp', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--samples_start', type=int, default=10000)
    parser.add_argument('--samples_test', type=int, default=10000)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ckpt_dir = os.path.join("ckpt", args.name)
    os.makedirs(ckpt_dir, exist_ok=True)

    test_dict = defaultdict(list)

    # Separate negative from positive
    samples = []
    samples += (glob.glob(os.path.join(args.data_dir, 'train', "*")))
    samples += (glob.glob(os.path.join(args.data_dir, 'test', "*")))

    csv_labels = []
    csv_labels += csv.DictReader(open(os.path.join(args.data_dir, "trainLabels.csv")), delimiter=',')
    csv_labels += csv.DictReader(open(os.path.join(args.data_dir, "valLabels.csv")), delimiter=',')

    positive = set([row['image'] for row in csv_labels if int(row['level']) > 0])
    negative = set([row['image'] for row in csv_labels if int(row['level']) == 0])

    teacher = resnet152()
    teacher.fc = nn.Linear(teacher.fc.in_features, 2 if args.task_binary else 5)
    teacher = nn.DataParallel(teacher)
    teacher.load_state_dict(torch.load('ckpt/teacher/model_best.pth')['state_dict'])
    teacher.cuda()
    teacher.eval()


    def loss_fn_kd(outputs, labels, teacher_outputs):
        """
        Compute the knowledge-distillation (KD) loss given outputs, labels.
        "Hyperparameters": temperature and alpha
        NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
        and student expects the input tensor to be log probabilities! See Issue #2
        """
        alpha = 0.5
        T = 1
        KD_loss = nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1),
                                 F.softmax(teacher_outputs / T, dim=1)) * (alpha * T * T) + \
                  F.cross_entropy(outputs, labels) * (1. - alpha)

        return KD_loss


    for num_exp in range(args.num_exp):

        samples_positive = sorted([s for s in samples if os.path.basename(os.path.splitext(s)[0]) in positive],
                                  key=lambda k: random.random())
        samples_negative = sorted([s for s in samples if os.path.basename(os.path.splitext(s)[0]) in negative],
                                  key=lambda k: random.random())

        print("Positive samples:", len(samples_positive),
              "Negative samples:", len(samples_negative))

        datas = []
        end_pos = 0
        end_neg = 0
        for i in range(args.rounds + 1):
            incr = args.samples_start if i == 0 else args.samples_test
            start_pos = end_pos
            start_neg = end_neg
            end_pos = int(start_pos + incr * 0.5)
            end_neg = int(start_neg + incr * 0.5)

            datas.append(sorted(samples_positive[start_pos:end_pos] + samples_negative[start_neg:end_neg],
                                key=lambda k: random.random()))

        net = eval(args.model)(pretrained=True)
        net.fc = nn.Linear(net.fc.in_features, 2 if args.task_binary else 5)
        net.cuda()

        criterion = nn.CrossEntropyLoss(reduction="sum").cuda()
        optimizer = optim.Adam(net.parameters(), lr=args.lr)

        for r in range(args.rounds):
            # Split current data round to train/val. Next round is test set
            eighty_percent = int(len(datas[r]) * 0.8)
            train_loader = DataLoader(Retina_Dataset_Simple(datas[r][:eighty_percent], csv_labels, args, mode='train'),
                                      args.batch_size, num_workers=4)
            val_loader = DataLoader(Retina_Dataset_Simple(datas[r][eighty_percent:], csv_labels, args, mode='val'),
                                    args.batch_size, num_workers=4)
            test_loader = DataLoader(Retina_Dataset_Simple(datas[r + 1], csv_labels, args, mode='test'),
                                     args.batch_size, num_workers=4)
            best_acc = 0
            early_stop = 0

            for epoch in range(args.epoch_per):
                net.train()
                for iteration, data in enumerate(train_loader):
                    inputs = data['image'].cuda()
                    labels = data['label'].cuda()
                    optimizer.zero_grad()

                    student_outputs = net(inputs)
                    teacher_outputs = teacher(inputs)

                    loss = loss_fn_kd(student_outputs, labels.flatten(), teacher_outputs)

                    loss.backward()
                    optimizer.step()

                    print("\r[Round %2d][Epoch %2d][Step %4d/%4d] Loss: %.4f, Lr: %.2e, ES: %4d" % (
                        r + 1,
                        epoch + 1,
                        iteration,
                        int(len(train_loader.dataset) / args.batch_size),
                        loss.cpu().data.numpy() / args.batch_size,
                        args.lr,  # *[group['lr'] for group in optim.param_groups],
                        early_stop,
                    ), end='          ')

                # Eval
                early_stop += 1
                net.eval()
                accuracy = []
                for iteration, data in enumerate(val_loader):
                    inputs = data['image'].cuda()
                    labels = data['label']
                    pred = net(inputs).cpu().data.numpy()
                    labels = labels.cpu().data.numpy()
                    accuracy += list(np.argmax(pred, axis=1) == labels.flatten())
                total_acc = 100 * np.mean(np.array(accuracy))
                print('Evaluation accuracy', str(total_acc))
                if total_acc > best_acc:
                    best_acc = total_acc
                    early_stop = 0
                    torch.save({
                        'round': r,
                        'state_dict': net.state_dict(),
                        'args': args,
                    },
                        os.path.join(ckpt_dir, 'model_best.pth'))
                if early_stop == args.early_stop:
                    break

            # Test
            # Load best eval model
            net.load_state_dict(torch.load(os.path.join(ckpt_dir, 'model_best.pth'))['state_dict'])
            net.eval()
            accuracy = []
            for iteration, data in enumerate(test_loader):
                inputs = data['image'].cuda()
                labels = data['label']
                pred = net(inputs).cpu().data.numpy()
                labels = labels.cpu().data.numpy()
                accuracy += list(np.argmax(pred, axis=1) == labels.flatten())
            test_acc = 100 * np.mean(np.array(accuracy))
            print('Test accuracy', str(test_acc))
            open(os.path.join(ckpt_dir, "test_log.txt"), 'a+').write(
                'Round ' + str(r + 1) + ': test acc is ' + str(test_acc) + '\n'
            )
            test_dict[r].append(test_acc)

    mean_str = '['
    for k, v in test_dict.items():
        mean_str += ('[' + str(k) + ',' + str(v) + ',' + str(np.mean(v)) + ',' + str(np.std(v)) + '],\n')
    mean_str += ']'
    print(mean_str)
    open(os.path.join(ckpt_dir, "test_log.txt"), 'a+').write(mean_str)
