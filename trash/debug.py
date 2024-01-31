from copy import deepcopy
import os
import sys
from numpy.lib.npyio import load

sys.path.insert(0, '/raid0/wjc/PFL-Seg-Trans/')
import argparse
import numpy as np

from scripts.tester_utils import eval_container
from utils.summary import create_logger, DisablePrint, create_summary
from utils.util import load_model

from scripts.train_fed import build_model
from glob import glob
import medpy.metric as mp


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fl', type=str, default='fedavg', help='exp_name')
    parser.add_argument('--net',
                        type=str,
                        default='pvtb0_fpn',
                        help='model_name')
    parser.add_argument('--ver', type=str, default='0', help='version')
    parser.add_argument('--dataset',
                        type=str,
                        default='polyp',
                        help='dataset name')

    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    args = parser.parse_args()

    args.exp = '{}_{}_{}'.format(args.fl, args.net, args.ver)
    args.txt_path = 'logs/{}/{}/txt/'.format(args.dataset, args.exp)
    args.log_path = 'logs/{}/{}/log/'.format(args.dataset, args.exp)
    args.model_path = 'logs/{}/{}/model/'.format(args.dataset, args.exp)
    args.npy_path = 'logs/{}/{}/npy/'.format(args.dataset, args.exp)
    os.makedirs(args.npy_path, exist_ok=True)
    args.logger = create_logger(0, save_dir=args.txt_path)
    print = args.logger.info

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
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
    args.ds = Dataset
    print(args)
    return args


if __name__ == "__main__":
    # define dataset, model, optimizer for each client
    import torch
    args = get_args()
    print = args.logger.info
    net = build_model(args)
    dataloader_clients = []
    net_clients = []
    optimizer_clients = []
    for client_idx in range(args.client_num):
        _net = deepcopy(net).cuda()
        _net = load_model(
            _net, args.model_path + '/Site{}_best.pth'.format(client_idx + 1))
        net_clients.append(_net)
    test_data_list = glob(
        '/raid/wjc/data/SpecializedFedSeg/polyp/Site5/test/image/*')

    all_scores = []
    for fid, filename in enumerate(test_data_list):
        data = np.load(filename)
        image = np.expand_dims(data[..., :3].transpose(2, 0, 1), axis=0) / 255.
        image = torch.from_numpy(image).cuda().float()
        mask = np.load(filename.replace('image', 'mask')) > 0.5

        print(image.shape)

        dice_scores = []
        preds = []
        for client_index in range(4):
            with torch.no_grad():
                pred = net_clients[client_index](image).cpu().numpy()[0, 0]
            if np.max(pred) <= 0.5:
                pred[0, 0] = 1
            preds.append(pred)
            dice_scores.append(mp.dc(pred > 0.5, mask))
        avg_pred = np.mean(preds, axis=0)
        dice_scores.append(mp.dc(avg_pred > 0.5, mask))
        all_scores.append(dice_scores)
        print(dice_scores)
    all_scores = np.array(all_scores)
    print(np.mean(all_scores, axis=0))
