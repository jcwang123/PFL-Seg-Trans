import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + '/..')

import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from torch.nn import KLDivLoss

from networks.simclr import MLP
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
    global_net = build_model(args)
    for client_idx in range(args.client_num):
        _net = deepcopy(global_net).to('cuda')
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
    # local_net_clients = deepcopy(net_clients)
    return dataloader_clients, optimizer_clients, net_clients


def train_one_ep(args,
                 dataloaders,
                 nets,
                 opts,
                 epoch_num,
                 writer,
                 w_ditto=None,
                 net_server=None,
                 mlps=None):
    print = args.logger.info
    kl_loss = KLDivLoss()
    local_update_clines = list(range(args.client_num))

    for client_idx in local_update_clines:
        dataloader_current = dataloaders[client_idx]
        net_current = nets[client_idx]
        net_current.train()
        net_previous = deepcopy(net_current)
        net_previous.eval()
        optimizer_current = opts[client_idx]

        for i_batch, sampled_batch in enumerate(dataloader_current):
            #------------------  obtain training data ------------------ #
            volume_batch, label_batch = sampled_batch['image'].cuda(
            ), sampled_batch['label'].cuda()

            #------------------  obtain updated parameter at inner loop ------------------ #
            if args.fl == 'fedgkd':
                with torch.no_grad():
                    output_server = net_server(volume_batch)
                outputs = net_current(volume_batch)
                loss1 = dice_loss(outputs, label_batch)
                loss2 = kl_loss(outputs, output_server)
                total_loss = loss1 + 0.1 * loss2
                # pass
            elif args.fl == 'moon':
                # todo
                pass
            else:
                if args.fl == 'fedrep':
                    pass
                elif args.fl == 'fedbabu':
                    freeze_params(net_current, keys=args.head_keys)
                elif args.fl == 'ditto':
                    w_0 = deepcopy(net_current.state_dict())

                outputs = net_current(volume_batch)
                total_loss = dice_loss(outputs, label_batch)

            optimizer_current.zero_grad()
            total_loss.backward()
            optimizer_current.step()

            if args.fl == 'ditto' and w_ditto is not None:
                w_net = deepcopy(net_current.state_dict())
                for key in w_net.keys():
                    w_net[key] = w_net[key] - args.base_lr * (w_0[key] -
                                                              w_ditto[key])
                net_current.load_state_dict(w_net)
                optimizer_current.zero_grad()

            #------------------ logger ------------------ #
            iter_num = len(dataloader_current) * epoch_num + i_batch
            if iter_num % 10 == 0 and writer is not None:
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
    dataloader_clients, opt_clients, net_clients = initial_trainer(args)
    if args.fl == 'ditto':
        net_servers = deepcopy(net_clients)
        opt_servers = [
            torch.optim.Adam(_net.parameters(), lr=args.base_lr)
            for _net in net_servers
        ]
    else:
        net_server = deepcopy(net_clients[0])

    # ------------------  decouple model parameters ------------------ #
    params = dict(net_clients[0].named_parameters())
    names = [name for name, param in params.items()]
    args.head_keys = list(filter(lambda x: 'p_head' in x, names))
    args.body_keys = list(filter(lambda x: 'p_head' not in x, names))
    print('Body params {} Head params {}'.format(len(args.body_keys),
                                                 len(args.head_keys)))

    # ------------------  build contrastive MLPs (MOON)------------------ #
    if args.fl == 'moon':
        mlps = [
            MLP(dim=256, projection_size=256, hidden_size=1024).cuda()
            for _ in range(args.client_num)
        ]
    else:
        mlps = None

    # ------------------  start federated training ------------------ #
    best_score = 0
    writer = SummaryWriter(args.log_path)
    for epoch_num in range(args.max_epoch):
        if args.fl == 'ditto':
            # ------------------  Ditto ------------------ #
            train_one_ep(args,
                         dataloader_clients,
                         net_servers,
                         opt_servers,
                         epoch_num,
                         writer=None)
            update_global_model(net_servers, args.client_weight)
            w_ditto = deepcopy(net_servers[0].state_dict())
            train_one_ep(args,
                         dataloader_clients,
                         net_clients,
                         opt_clients,
                         epoch_num,
                         writer=writer,
                         w_ditto=w_ditto)
        else:
            # ------------------  (Fedavg, FedGKD), (FedRep, FedBABU) ------------------ #
            train_one_ep(args,
                         dataloader_clients,
                         net_clients,
                         opt_clients,
                         epoch_num,
                         writer,
                         net_server=net_server,
                         mlps=mlps)

            if args.fl in ['fedavg', 'fedgkd', 'moon']:
                update_global_model(net_clients, args.client_weight)
                if args.fl == 'moon':
                    update_global_model(mlps, args.client_weight)
                net_server = deepcopy(net_clients[0])
                net_server.eval()

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
