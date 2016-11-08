# --------------------------------------------------------
# SubCNN_TF
# Copyright (c) 2016 CVGL Stanford
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import networks.VGG_vid_train
import networks.VGG_vid_test
import networks.VGG_pascal_train
import networks.VGG_pascal_test
import pdb
import tensorflow as tf

__sets['VGG_vid_train'] = networks.VGG_vid_train
__sets['VGG_vid_test'] = networks.VGG_vid_test
__sets['VGG_pascal_train'] = networks.VGG_pascal_train
__sets['VGG_pascal_test'] = networks.VGG_pascal_test


def get_network(name):
    """Get a network by name."""
    try:
        net = __sets[name]
    except:
        raise KeyError('Unknown network name: {}'.format(name))
    return net()
    

def list_networks():
    """List all registered imdbs."""
    return __sets.keys()
