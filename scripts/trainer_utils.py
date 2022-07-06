import numpy as np
import torch
from torch.autograd import Variable


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def set_global_grad(net, keys, tag):
    for name, param in net.named_parameters():
        if name in keys:
            param.requires_grad = (tag == 1)
        else:
            param.requires_grad = (tag == 0)


def check_equal(net_clients):
    params = dict(net_clients[0].named_parameters())
    for name, param in params.items():
        for client in range(1, len(net_clients)):
            _tmp_param_data = dict(
                net_clients[client].named_parameters())[name].data
            assert torch.sum(_tmp_param_data - params[name].data) == 0


def freeze_params(net, keys):
    params = dict(net.named_parameters())
    for name, param in params.items():
        if name in keys:
            dict(net.named_parameters())[name].requires_grad = False
        else:
            dict(net.named_parameters())[name].requires_grad = True


def update_global_model(net_clients, client_weight):
    print('Calculate the model avg----')
    params = dict(net_clients[0].named_parameters())
    for name, param in params.items():
        for client in range(len(net_clients)):
            single_client_weight = client_weight[client]
            if client == 0:
                tmp_param_data = dict(net_clients[client].named_parameters()
                                      )[name].data * single_client_weight
            else:
                tmp_param_data = tmp_param_data + \
                                 dict(net_clients[client].named_parameters())[
                                     name].data * single_client_weight
            params[name].data.copy_(tmp_param_data)
    print('Update each client model parameters----')

    for client in range(len(net_clients)):
        tmp_params = dict(net_clients[client].named_parameters())
        for name, param in params.items():
            tmp_params[name].data.copy_(param.data)


def update_global_model_with_keys(net_clients, client_weight, private_keys):
    print('Calculate the model avg----')
    params = dict(net_clients[0].named_parameters())
    for name, param in params.items():
        for client in range(len(net_clients)):
            single_client_weight = client_weight[client]
            if client == 0:
                tmp_param_data = dict(net_clients[client].named_parameters()
                                      )[name].data * single_client_weight
            else:
                tmp_param_data = tmp_param_data + \
                                 dict(net_clients[client].named_parameters())[
                                     name].data * single_client_weight
            params[name].data.copy_(tmp_param_data)
    print('Update each client model parameters----')

    for client in range(len(net_clients)):
        tmp_params = dict(net_clients[client].named_parameters())
        for name, param in params.items():
            if name in private_keys:
                print('Ignore param: {}'.format(name))
                continue
            tmp_params[name].data.copy_(param.data)


def update_global_model_for_trans(net_clients, client_weight):
    # 'kv'
    print('Calculate the model avg----')
    params = dict(net_clients[0].named_parameters())
    for name, param in params.items():
        for client in range(len(net_clients)):
            single_client_weight = client_weight[client]
            if client == 0:
                tmp_param_data = dict(net_clients[client].named_parameters()
                                      )[name].data * single_client_weight
            else:
                tmp_param_data = tmp_param_data + \
                                 dict(net_clients[client].named_parameters())[
                                     name].data * single_client_weight
            params[name].data.copy_(tmp_param_data)
    print('Update each client model parameters----')

    for client in range(len(net_clients)):
        tmp_params = dict(net_clients[client].named_parameters())
        for name, param in params.items():
            if 'kv' in name:
                print('save half the param: {}'.format(name))
                l = param.data.size()[0]
                new = torch.cat(
                    [param.data[:l // 2], tmp_params[name].data[l // 2:]],
                    dim=0)
                tmp_params[name].data.copy_(new)

                # print(tmp_params[name][0], tmp_params[name][-1],
                #   param.data[:l // 2][0], param.data[:l // 2][-1],
                #   tmp_params[name].data[0], tmp_params[name].data[-1])
            else:
                tmp_params[name].data.copy_(param.data)