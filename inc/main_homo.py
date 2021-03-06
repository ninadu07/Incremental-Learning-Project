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

from retina_dataset_simple import Retina_Dataset_Simple
import pandas as pd

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path = 'Config')
def homo(cfg: DictConfig): 
    # Seed
    if cfg.seed == None: 
        cfg.seed = random.randint(0, 9999999)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ckpt_dir = os.path.join("ckpt", cfg.name)
    os.makedirs(ckpt_dir, exist_ok=True)

    test_dict = defaultdict(list)

    # Separate negative from positive
    samples = []
    samples += (glob.glob(os.path.join(cfg.data_dir, 'train', "*")))
    samples += (glob.glob(os.path.join(cfg.data_dir, 'test', "*")))

    csv_labels = []
    csv_labels += csv.DictReader(open(os.path.join(cfg.data_dir, "trainLabels.csv")), delimiter=',')
    csv_labels += csv.DictReader(open(os.path.join(cfg.data_dir, "valLabels.csv")), delimiter=',')

    positive = set([row['image'] for row in csv_labels if int(row['level']) > 0])
    negative = set([row['image'] for row in csv_labels if int(row['level']) == 0])

    for num_exp in range(cfg.num_exp):

        samples_positive = sorted([s for s in samples if os.path.basename(os.path.splitext(s)[0]) in positive],
                                  key=lambda k: random.random())
        samples_negative = sorted([s for s in samples if os.path.basename(os.path.splitext(s)[0]) in negative],
                                  key=lambda k: random.random())

        print("Positive samples:", len(samples_positive),
              "Negative samples:", len(samples_negative))

        datas = []
        end_pos = 0
        end_neg = 0
        for i in range(cfg.rounds+1):
            incr = cfg.samples_start if i == 0 else cfg.samples_test
            start_pos = end_pos
            start_neg = end_neg
            end_pos = int(start_pos + incr * cfg.train_dist[i])
            end_neg = int(start_neg + incr * (1-cfg.train_dist[i]))

            datas.append(sorted(samples_positive[start_pos:end_pos] + samples_negative[start_neg:end_neg],
                                key=lambda k: random.random()))

        net = eval(cfg.model)(pretrained=True)
        net.fc = nn.Linear(net.fc.in_features, 2 if cfg.task_binary else 5)
        net.cuda()
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            net = nn.DataParallel(net)
        criterion = nn.CrossEntropyLoss(reduction="sum").cuda()
        optimizer = optim.Adam(net.parameters(), lr=cfg.lr)

        for r in range(cfg.rounds):
            # Split current data round to train/val. Next round is test set
            eighty_percent = int(len(datas[r]) * 0.8)
            train_loader = DataLoader(Retina_Dataset_Simple(datas[r][:eighty_percent], csv_labels, cfg, mode='train'),
                                      cfg.batch_size, num_workers=4)
            val_loader = DataLoader(Retina_Dataset_Simple(datas[r][eighty_percent:], csv_labels, cfg, mode='val'),
                                    cfg.batch_size, num_workers=4)
            test_loader = DataLoader(Retina_Dataset_Simple(datas[r + 1], csv_labels, cfg, mode='test'),
                                     cfg.batch_size, num_workers=4)
            best_acc = 0
            early_stop = 0

            for epoch in range(cfg.epoch_per):
                net.train()
                for iteration, data in enumerate(train_loader):
                    inputs = data['image'].cuda()
                    labels = data['label'].cuda()
                    optimizer.zero_grad()

                    outputs = net(inputs)

                    loss = criterion(outputs, labels.flatten())
                    loss.backward()
                    optimizer.step()

                    print("\r[Round %2d][Epoch %2d][Step %4d/%4d] Loss: %.4f, Lr: %.2e, ES: %4d" % (
                        r + 1,
                        epoch + 1,
                        iteration,
                        int(len(train_loader.dataset) / cfg.batch_size),
                        loss.cpu().data.numpy() / cfg.batch_size,
                        cfg.lr,  # *[group['lr'] for group in optim.param_groups],
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
                        'cfg': cfg,
                    },
                        os.path.join(ckpt_dir, 'model_best.pth'))
                if early_stop == cfg.early_stop:
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

if __name__ == '__main__':
    homo()