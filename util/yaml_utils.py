# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Copied from https://github.com/pfnet-research/sngan_projection/blob/master/source/yaml_utils.py

import argparse
import os
import shutil
import sys
import time

import yaml


class Config(object):
    def __init__(self, config_dict):
        self.config = config_dict

    def __getattr__(self, key):
        if key in self.config:
            return self.config[key]
        else:
            raise AttributeError(key)

    def __getitem__(self, key):
        return self.config[key]

    def __repr__(self):
        return yaml.dump(self.config, default_flow_style=False)


def load_dataset(config, test=False):
    dataset = load_module(config.dataset['dataset_func'],
                          config.dataset['dataset_name'])
    args = config.dataset['args']
    args['test'] = test
    return dataset(**args)


def load_module(func, name):
    mod_name = os.path.splitext(os.path.basename(func))[0]
    mod_path = os.path.dirname(func)
    sys.path.insert(0, mod_path)
    return getattr(__import__(mod_name), name)


def load_model(model_func, model_name, args=None):
    model = load_module(model_func, model_name)
    if args:
        return model(**args)
    return model()


def load_optimizer(opt_name, args=None):
    opt_module = __import__('chainer.optimizers', fromlist=[opt_name])
    opt_class = getattr(opt_module, opt_name)
    if args:
        return opt_class(**args)
    return opt_class()


def load_updater_class(config):
    func = config.updater['func']
    name = config.updater['name']
    if func.endswith('.py'):
        updater = load_module(func, name)
    else:
        updater = getattr(__import__(func, fromlist=[name]), name)
    return updater


def load_preupdater_class(config):
    func = config.preupdater['func']
    name = config.preupdater['name']
    if func.endswith('.py'):
        preupdater = load_module(func, name)
    else:
        preupdater = getattr(__import__(func, fromlist=[name]), name)
    return preupdater


def load_method(method_config):
    func = method_config['func']
    name = method_config['name']
    if func.endswith('.py'):
        method = load_module(func, name)
    else:
        method = getattr(__import__(func, fromlist=[name]), name)
    return method


def load_method_concrete(func, name):
    if func.endswith('.py'):
        method = load_module(func, name)
    else:
        method = getattr(__import__(func, fromlist=[name]), name)
    return method
