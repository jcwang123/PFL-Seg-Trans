import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + '/..')

import torch
import numpy as np
import random
from torch.utils.data import DataLoader

from scripts.trainer_utils import set_global_grad, update_global_model, update_global_model_with_keys, check_equal, freeze_params
from scripts.tester_utils import eval_container
from utils.losses import dice_loss
from config.base import get_args

from tensorboardX import SummaryWriter
from copy import deepcopy


def build_model(args):
    from networks.FPN.model import BuildFPN
    encoder = args.net.split('_')[0]
    decoder = args.net.split('_')[1]
    print("build net with encoder {} and decoder {}.".format(encoder, decoder))
    net = BuildFPN(args.num_classes, encoder, decoder)
    return net


def initial_trainer(args):
    # define dataset, model, optimizer for each client
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    dataloader_clients = []
    net_clients = []
    optimizer_clients = []
    net = build_model(args)
    for client_idx in range(args.client_num):
        _net = deepcopy(net).to('cuda')

        dataset = args.ds_func(client_idx=client_idx, split='train')
        dataloader = DataLoader(dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=1,
                                pin_memory=True,
                                worker_init_fn=worker_init_fn)
        dataloader_clients.append(dataloader)

        optimizer = torch.optim.Adam(_net.parameters(),
                                     lr=args.base_lr,
                                     betas=(0.9, 0.999))
        optimizer_clients.append(optimizer)
        net_clients.append(_net)

        print('[INFO] Initialized success...')
    return dataloader_clients, optimizer_clients, net_clients


def train_one_ep(args, dataloader_clients, net_clients, optimizer_clients,
                 epoch_num, writer):
    print = args.logger.info
    local_update_clines = list(range(args.client_num))

    for client_idx in local_update_clines:
        dataloader_current = dataloader_clients[client_idx]
        net_current = net_clients[client_idx]
        net_current.train()
        optimizer_current = optimizer_clients[client_idx]

        for i_batch, sampled_batch in enumerate(dataloader_current):
            #------------------  obtain training data ------------------ #
            volume_batch, label_batch = sampled_batch['image'].cuda(
            ), sampled_batch['label'].cuda()

            #------------------  obtain updated parameter at inner loop ------------------ #
            if args.fl == 'fedrep':
                if i_batch <= 10:
                    freeze_params(net_current, keys=args.body_keys)
                else:
                    freeze_params(net_current, keys=args.head_keys)
            elif args.fl == 'fedbabu':
                freeze_params(net_current, keys=args.head_keys)

            outputs = net_current(volume_batch)
            total_loss = dice_loss(outputs, label_batch)

            optimizer_current.zero_grad()
            total_loss.backward()
            optimizer_current.step()

            #------------------ logger ------------------ #
            iter_num = len(dataloader_current) * epoch_num + i_batch
            if iter_num % 10 == 0:
                writer.add_scalar('loss/site{}'.format(client_idx + 1),
                                  total_loss, iter_num)
                print(
                    'Epoch: [%d] client [%d] iteration [%d / %d] : total loss : %f'
                    % (epoch_num, client_idx, iter_num,
                       len(dataloader_current), total_loss.item()))


def main():
    # ------------------  create args ------------------ #
    args = get_args()
    print = args.logger.info
    print(args)

    # ------------------  initialize the trainer ------------------ #
    dataloader_clients, optimizer_clients, net_clients = initial_trainer(args)

    # ------------------  decouple model parameters ------------------ #
    params = dict(net_clients[0].named_parameters())
    names = [name for name, param in params.items()]
    args.head_keys = list(filter(lambda x: 'p_head' in x, names))
    args.body_keys = list(filter(lambda x: 'p_head' not in x, names))
    print('Body params {} Head params {}'.format(len(args.body_keys),
                                                 len(args.head_keys)))

    # ------------------  start federated training ------------------ #
    best_score = 0
    writer = SummaryWriter(args.log_path)
    for epoch_num in range(args.max_epoch):
        # ------------------  local ------------------ #
        train_one_ep(args, dataloader_clients, net_clients, optimizer_clients,
                     epoch_num, writer)

        # ------------------  server ------------------ #
        if args.fl == 'fedavg':
            update_global_model(net_clients, args.client_weight)
        elif args.fl in ['fedrep', 'fedbabu']:
            update_global_model_with_keys(net_clients,
                                          args.client_weight,
                                          private_keys=args.head_keys)
        else:
            raise NotImplementedError

        # ------------------  evaluation ------------------ #
        overall_score = 0
        for site_index in range(args.client_num):
            this_net = net_clients[site_index]
            print("[Test] epoch {} testing Site {}".format(
                epoch_num, site_index + 1))

            score_values = eval_container(site_index, this_net, args)
            writer.add_scalar('Score/site{}'.format(site_index + 1),
                              np.mean(score_values[0]), epoch_num)
            overall_score += np.mean(score_values[0])
        overall_score /= args.client_num
        writer.add_scalar('Score_Overall', overall_score, epoch_num)

        # ------------------  save model ------------------ #
        if overall_score > best_score:
            best_score = overall_score
            # save model
            save_mode_path = os.path.join(args.model_path, 'best.pth')
            torch.save(net_clients[0].state_dict(), save_mode_path)

            for site_index in range(args.client_num):
                save_mode_path = os.path.join(
                    args.model_path, 'Site{}_best.pth'.format(site_index + 1))
                torch.save(net_clients[site_index].state_dict(),
                           save_mode_path)
        print('[INFO] Dice Overall: {:.2f} Best Dice {:.2f}'.format(
            overall_score * 100, best_score * 100))


if __name__ == "__main__":
    main()
