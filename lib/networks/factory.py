# --------------------------------------------------------
# SubCNN_TF
# Copyright (c) 2016 CVGL Stanford
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from networks.VGG_vid_train import VGG_vid_train
from networks.VGG_vid_test import VGG_vid_test
from networks.VGG_pascal_train import VGG_pascal_train
from networks.VGG_pascal_test import VGG_pascal_test
from networks.VGG_pascal_test_lowres import VGG_pascal_test_lowres
import pdb
import tensorflow as tf

__sets['VGG_vid_train'] = VGG_vid_train
__sets['VGG_vid_test'] = VGG_vid_test
__sets['VGG_pascal_train'] = VGG_pascal_train
__sets['VGG_pascal_test'] = VGG_pascal_test
__sets['VGG_pascal_test_lowres'] = VGG_pascal_test_lowres


def get_network(name):
    """Get a network by name."""
    try:
        net = __sets[name]
    except:
        print list_networks()
        raise KeyError('Unknown network name: {}'.format(name))
    return net()
    

def list_networks():
    """List all registered imdbs."""
    return __sets.keys()
