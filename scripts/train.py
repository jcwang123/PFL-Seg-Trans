from copy import deepcopy
import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + '/..')
from tqdm import tqdm
from tensorboardX import SummaryWriter

import argparse

import time
import random
import numpy as np

from glob import glob

import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader

from config.base import get_args

from utils.losses import dice_loss
from utils.summary import create_logger, DisablePrint, create_summary

from scripts.tester_utils import eval_container
from scripts.trainer_utils import set_global_grad, update_global_model, update_global_model_with_keys, check_equal


def build_model(args):
    from networks.FPN.model import BuildFPN
    encoder = args.net.split('_')[0]
    decoder = args.net.split('_')[1]
    print("build net with encoder {} and decoder {}.".format(encoder, decoder))
    net = BuildFPN(args.num_classes, encoder, decoder)


def initial_trainer(args, Dataset):
    client_weight = np.ones((args.client_num, )) / args.client_num
    print(client_weight)

    # define dataset, model, optimizer for each client
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    dataloader_clients = []
    net_clients = []
    optimizer_clients = []
    net = build_model(args)
    for client_idx in range(args.client_num):
        _net = deepcopy(net).to('cuda')

        dataset = Dataset(client_idx=client_idx, split='train')
        dataloader = DataLoader(dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=1,
                                pin_memory=True,
                                worker_init_fn=worker_init_fn)
        dataloader_clients.append(dataloader)

        optimizer = torch.optim.Adam(net.parameters(),
                                     lr=args.base_lr,
                                     betas=(0.9, 0.999))
        optimizer_clients.append(optimizer)
        net_clients.append(_net)

        print('[INFO] Initialized success...')
    return dataloader_clients, optimizer_clients, net_clients


def main():
    # ------------------  create args ------------------ #
    args = get_args()
    assert (args.load_weight is not None)
    args.exp = '{}_{}_{}'.format(args.fl, args.net, args.ver)
    txt_path = 'logs/{}/{}/txt/'.format(args.dataset, args.exp)
    log_path = 'logs/{}/{}/log/'.format(args.dataset, args.exp)
    model_path = 'logs/{}/{}/model/'.format(args.dataset, args.exp)
    os.makedirs(txt_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    logger = create_logger(0, save_dir=txt_path)
    print = logger.info

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.batch_size = args.batch_size * len(args.gpu.split(','))

    if args.dataset == 'fundus':
        from dataloaders.fundus_dataloader import Dataset, RandomNoise
        args.client_num = 4
        args.num_classes = 2
        args.c_in = 3
    elif args.dataset == 'pmr':
        from dataloaders.pmr_dataloader import Dataset, RandomNoise
        args.client_num = 6
        args.num_classes = 1
        args.c_in = 1
    elif args.dataset == 'polyp':
        from dataloaders.polyp_dataloader import Dataset, RandomNoise
        args.client_num = 4
        args.num_classes = 1
        args.c_in = 3
    else:
        raise NotImplementedError
    assert args.num_classes > 0 and args.client_num > 1
    print(args)

    # ------------------  start training ------------------ #
    dataloader_clients, optimizer_clients, net_clients = initial_trainer(
        args, Dataset)
    # weight average

    # ------------------  start federated training ------------------ #
    best_score = np.zeros((args.client_num, ))
    writer = SummaryWriter(log_path)
    for epoch_num in range(args.max_epoch):
        for client_idx in range(args.client_num):
            dataloader_current = dataloader_clients[client_idx]
            net_current = net_clients[client_idx]
            net_current.train()
            optimizer_current = optimizer_clients[client_idx]

            for i_batch, sampled_batch in enumerate(dataloader_current):
                # obtain training data
                volume_batch, label_batch = sampled_batch['image'].cuda(
                ), sampled_batch['label'].cuda()

                # obtain updated parameter at inner loop
                outputs = net_current(volume_batch)

                total_loss = dice_loss(outputs, label_batch)

                optimizer_current.zero_grad()
                total_loss.backward()
                optimizer_current.step()

                iter_num = len(dataloader_current) * epoch_num + i_batch
                if iter_num % 10 == 0:
                    writer.add_scalar('loss/site{}'.format(client_idx + 1),
                                      total_loss, iter_num)
                    print(
                        'Epoch: [%d] client [%d] iteration [%d / %d] : total loss : %f'
                        % (epoch_num, client_idx, iter_num,
                           len(dataloader_current), total_loss.item()))

        for site_index in range(args.client_num):
            this_net = net_clients[site_index]
            this_net.eval()
            print("[Test] epoch {} testing Site {}".format(
                epoch_num, site_index + 1))

            score_values = eval_container(site_index, this_net, args)
            score = np.mean(score_values[0])
            if score > best_score[site_index]:
                best_score[site_index] = score
                save_mode_path = os.path.join(
                    model_path, 'Site{}_best.pth'.format(site_index + 1))
                torch.save(net_clients[site_index].state_dict(),
                           save_mode_path)
            print('[INFO] IoU score {:.4f} Best score {:.4f}'.format(
                np.mean(score_values[0]), best_score[site_index]))


if __name__ == "__main__":
    main()