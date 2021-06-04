# ------------------------------------------------------------------------
# HOTR official code : hotr/util/logger.py
# Copyright (c) Kakao Brain, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch
import time
import datetime
import sys
from time import sleep
from collections import defaultdict

from hotr.util.misc import SmoothedValue

def print_params(model):
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\n[Logger] Number of params: ', n_parameters)
    return n_parameters

def print_args(args):
    print('\n[Logger] DETR Arguments:')
    for k, v in vars(args).items():
        if k in [
            'lr', 'lr_backbone', 'lr_drop',
            'frozen_weights',
            'backbone', 'dilation',
            'position_embedding', 'enc_layers', 'dec_layers', 'num_queries',
            'dataset_file']:
            print(f'\t{k}: {v}')

    if args.HOIDet:
        print('\n[Logger] DETR_HOI Arguments:')
        for k, v in vars(args).items():
            if k in [
                'freeze_enc',
                'query_flag',
                'hoi_nheads',
                'hoi_dim_feedforward',
                'hoi_dec_layers',
                'hoi_idx_loss_coef',
                'hoi_act_loss_coef',
                'hoi_eos_coef',
                'object_threshold']:
                print(f'\t{k}: {v}')

class MetricLogger(object):
    def __init__(self, mode="test", delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.mode = mode

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)

            if (i % print_freq == 0 and i !=0) or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i+1, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB),
                        flush=(self.mode=='test'), end=("\r" if self.mode=='test' else "\n"))
                else:
                    print(log_msg.format(
                        i+1, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)),
                        flush=(self.mode=='test'), end=("\r" if self.mode=='test' else "\n"))
            else:
                log_interval = self.delimiter.join([header, '[{0' + space_fmt + '}/{1}]'])
                if torch.cuda.is_available(): print(log_interval.format(i+1, len(iterable)), flush=True, end="\r")
                else: print(log_interval.format(i+1, len(iterable)), flush=True, end="\r")

            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if self.mode=='test': print("")
        print('[stats] Total Time ({}) : {} ({:.4f} s / it)'.format(
            self.mode, total_time_str, total_time / len(iterable)))
