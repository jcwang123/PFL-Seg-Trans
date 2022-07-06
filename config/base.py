import argparse
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(__file__) + '/..')
from utils.summary import create_logger, DisablePrint, create_summary


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fl', type=str, default='fedavg', help='exp_name')
    parser.add_argument('--net', type=str, default='unet', help='model_name')
    parser.add_argument('--ver', type=str, default='0', help='version')
    parser.add_argument('--dataset',
                        type=str,
                        default='polyp',
                        help='dataset name')
    parser.add_argument('--max_epoch',
                        type=int,
                        default=200,
                        help='maximum epoch number to train')
    parser.add_argument('--batch_size',
                        type=int,
                        default=8,
                        help='batch_size per gpu')
    parser.add_argument('--base_lr',
                        type=float,
                        default=0.001,
                        help='basic learning rate of each site')
    parser.add_argument('--load_weight',
                        type=np.str,
                        default=None,
                        help='load pre-trained weight from local site')

    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument('--gpu', type=str, default='1', help='GPU to use')
    args = parser.parse_args()
    args.exp = '{}_{}_{}'.format(args.fl, args.net, args.ver)
    args.txt_path = 'logs/{}/{}/txt/'.format(args.dataset, args.exp)
    args.log_path = 'logs/{}/{}/log/'.format(args.dataset, args.exp)
    args.model_path = 'logs/{}/{}/model/'.format(args.dataset, args.exp)
    os.makedirs(args.txt_path, exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)
    args.logger = create_logger(0, save_dir=args.txt_path)

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
    args.client_weight = np.ones((args.client_num, )) / args.client_num
    args.ds_func = Dataset
    return args