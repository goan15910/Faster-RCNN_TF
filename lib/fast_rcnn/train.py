# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

from fast_rcnn.config import cfg
import gt_data_layer.roidb as gdl_roidb
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.vid_layer import RoIDataLayer
from utils.timer import Timer
import numpy as np
import os
import tensorflow as tf
import sys


class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, sess, network, imdb, roidb, output_dir, pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.net = network
        self.imdb = imdb
        self.roidb = roidb
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model

        print 'Computing bounding-box regression targets...'
        if cfg.TRAIN.BBOX_REG:
            self.bbox_means, self.bbox_stds = rdl_roidb.add_bbox_regression_targets(roidb)
        print 'done'

        # For checkpoint
        self.saver = tf.train.Saver(max_to_keep=100)

    def snapshot(self, sess, iter):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.net

        if cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred'):
            # save original values
            with tf.variable_scope('bbox_pred', reuse=True):
                weights = tf.get_variable("weights")
                biases = tf.get_variable("biases")

            orig_0 = weights.eval()
            orig_1 = biases.eval()

            # scale and shift with bbox reg unnormalization; then save snapshot
            weights_shape = weights.get_shape().as_list()
            sess.run(weights.assign(orig_0 * np.tile(self.bbox_stds, (weights_shape[0],1))))
            sess.run(biases.assign(orig_1 * self.bbox_stds + self.bbox_means))

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
                    '_iter_{:d}'.format(iter+1) + '.ckpt')
        filename = os.path.join(self.output_dir, filename)

        self.saver.save(sess, filename)
        print 'Wrote snapshot to: {:s}'.format(filename)

        if cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred'):
            # restore net to original state
            sess.run(weights.assign(orig_0))
            sess.run(biases.assign(orig_1))

    def rpn_cls_loss(self):
        # RPN classification loss
        name = "rpn_cls_loss"
        rpn_cls_score = tf.reshape(self.net.get_output('rpn_cls_score_reshape'),[-1,2])
        rpn_label = tf.reshape(self.net.get_output('rpn-data')[0],[-1])
        # ignore_label(-1)
        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score,tf.where(tf.not_equal(rpn_label,-1))),[-1,2])
        rpn_label = tf.reshape(tf.gather(rpn_label,tf.where(tf.not_equal(rpn_label,-1))),[-1])
        rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(rpn_cls_score, rpn_label), name=name)
        return rpn_cross_entropy

    def rpn_bbox_loss(self):
        # RPN bounding box regression L1 loss
        name = "rpn_bbox_loss"
        rpn_bbox_pred = self.net.get_output('rpn_bbox_pred')
        rpn_bbox_targets = tf.transpose(self.net.get_output('rpn-data')[1],[0,2,3,1])
        rpn_bbox_inside_weights = tf.transpose(self.net.get_output('rpn-data')[2],[0,2,3,1])
        rpn_bbox_outside_weights = tf.transpose(self.net.get_output('rpn-data')[3],[0,2,3,1])
        smoothL1_sign = tf.cast(tf.less(tf.abs(tf.sub(rpn_bbox_pred, rpn_bbox_targets)),1),tf.float32)
        rpn_loss_box = tf.mul(tf.reduce_mean(tf.reduce_sum(tf.mul(rpn_bbox_outside_weights,tf.add(
                       tf.mul(tf.mul(tf.pow(tf.mul(rpn_bbox_inside_weights, tf.sub(rpn_bbox_pred, rpn_bbox_targets))*3,2),0.5),smoothL1_sign),
                       tf.mul(tf.sub(tf.abs(tf.sub(rpn_bbox_pred, rpn_bbox_targets)),0.5/9.0),tf.abs(smoothL1_sign-1)))), reduction_indices=[1,2])),10, name=name)
        return rpn_loss_box

    def rcnn_cls_loss(self):
        # RCNN classification loss
        name = "rcnn_cls_loss"
        cls_score = self.net.get_output('cls_score')
        label = tf.reshape(self.net.get_output('roi-data')[1],[-1])
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(cls_score, label), name=name)
        return cross_entropy

    def rcnn_bbox_loss(self):
        # RCNN bounding box regression L1 loss
        name = "rcnn_bbox_loss"
        bbox_pred = self.net.get_output('bbox_pred')
        bbox_targets = self.net.get_output('roi-data')[2]
        bbox_inside_weights = self.net.get_output('roi-data')[3]
        bbox_outside_weights = self.net.get_output('roi-data')[4]
        loss_box = tf.reduce_mean(tf.reduce_sum(tf.mul(bbox_outside_weights,tf.mul(bbox_inside_weights, tf.abs(tf.sub(bbox_pred, bbox_targets)))), reduction_indices=[1]), name=name)
        return loss_box

    def _add_losses_summaries(self, total_loss, loss_set=None):
        if loss_set is not None:
            losses = tf.get_collection(loss_set) + [total_loss]
        else:
            losses = [total_loss]
        for l in losses:
            tf.scalar_summary(l.op.name, l)

    def train_model(self, sess, max_iters):
        """Network training loop."""

        data_layer = get_data_layer(self.roidb, self.imdb.num_classes)

        # Set objective loss
        obj_loss_set = "objective_losses"
        tf.add_to_collection(obj_loss_set, self.rpn_cls_loss())
        tf.add_to_collection(obj_loss_set, self.rpn_bbox_loss())
        tf.add_to_collection(obj_loss_set, self.rcnn_cls_loss())
        tf.add_to_collection(obj_loss_set, self.rcnn_bbox_loss())
        obj_loss = tf.add_n(tf.get_collection(obj_loss_set), name='objective_loss')

        if cfg.TRAIN.WEIGHT_DECAY:
            # weight-decay L2 loss
            wd_loss_set = "weight_decay_losses"
            wd = cfg.TRAIN.WEIGHT_DECAY_FACTOR
            for var in self.net.get_trainable_weights():
                weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
                tf.add_to_collection(wd_loss_set, weight_decay)
            wd_loss = tf.add_n(tf.get_collection(wd_loss_set), name='total_weight_loss')
            total_loss = tf.add(obj_loss, wd_loss, name="total_loss")
        else:
            total_loss = tf.add(obj_loss, 0, name="total_loss")

        # optimizer
        lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
        momentum = cfg.TRAIN.MOMENTUM
        train_op = tf.train.MomentumOptimizer(lr, momentum).minimize(total_loss)

        # summary writer
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(self.output_dir, sess.graph)

        # iintialize variables
        sess.run(tf.initialize_all_variables())
        if self.pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(self.pretrained_model)
            self.net.load(self.pretrained_model, sess, True)

        last_snapshot_iter = -1
        timer = Timer()
        for iter in range(max_iters):
            # Learning rate
            if iter >= cfg.TRAIN.STEPSIZE:
                sess.run(tf.assign(lr, cfg.TRAIN.LEARNING_RATE * cfg.TRAIN.GAMMA))
            else:
                sess.run(tf.assign(lr, cfg.TRAIN.LEARNING_RATE))
            
            # Add loss summaries
            self._add_losses_summaries(total_loss, obj_loss_set)

            # Get one batch
            blobs = data_layer.forward()

            # Make one SGD update
            feed_dict={self.net.data: blobs['data'], self.net.im_info: blobs['im_info'], self.net.keep_prob: 0.5, \
                           self.net.gt_boxes: blobs['gt_boxes']}
            timer.tic()

            rpn_cls, rpn_bbox, rcnn_cls, rcnn_bbox, _ = sess.run(tf.get_collection(obj_loss_set) + [train_op], feed_dict=feed_dict)

            timer.toc()

            if (iter+1) % (cfg.TRAIN.DISPLAY) == 0:
                total_loss_value = rpn_cls + rpn_bbox + rcnn_cls + rcnn_bbox
                print 'iter: %d / %d,total loss: %.4f, rpn_loss_cls: %.4f, rpn_loss_box: %.4f, loss_cls: %.4f, loss_box: %.4f, lr: %f'%\
                        (iter+1, max_iters, total_loss_value, rpn_cls, rpn_bbox, rcnn_cls, rcnn_bbox, lr.eval())
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if (iter+1) % cfg.TRAIN.SUMMARY_ITERS == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, iter)

            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    print 'Preparing training data...'
    if cfg.TRAIN.HAS_RPN:
        if cfg.IS_MULTISCALE:
            gdl_roidb.prepare_roidb(imdb)
        else:
            rdl_roidb.prepare_roidb(imdb)
    else:
        rdl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb


def get_data_layer(roidb, num_classes):
    """return a data layer."""
    if cfg.TRAIN.HAS_RPN:
        if cfg.IS_MULTISCALE:
            layer = GtDataLayer(roidb)
        else:
            layer = RoIDataLayer(roidb, num_classes)
    else:
        layer = RoIDataLayer(roidb, num_classes)

    return layer


def train_net(network, imdb, roidb, output_dir, pretrained_model=None, max_iters=40000):
    """Train a Fast R-CNN network."""

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sw = SolverWrapper(sess, network, imdb, roidb, output_dir, pretrained_model=pretrained_model)
        print 'Solving...'
        sw.train_model(sess, max_iters)
        print 'done solving'
