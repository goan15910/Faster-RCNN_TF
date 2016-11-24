import sys,os
import os.path as osp
import numpy as np
import scipy
import scipy.io
import argparse

import selectivesearch


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Sample frames from original dataset')
    parser.add_argument('--imgs', dest='img_fname', help='Absolute path of the dataset imagefile', type=str)
    parser.add_argument('--keepid', dest='keep_ids', help='Keep indexes file', default=None, type=str)
    parser.add_argument('--method', dest='method', help='Algorithm to generate bboxes', default='selectvie_search', type=str)
    parser.add_argument('--ext', dest='ext', help='Extension of image files', default="JPEG", type=str)
    parser.add_argument('--opath', dest='opath', help='Output path', default=".", type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def selective_search(img_fname, params, keep_size=2000):
    """
    Run Selective Search on the given image filenames to
    generate bbox proposals.

    Returns:
    A ndarray of bboxes of the image
    """
    img = scipy.misc.imread(img_fname)
    img_lbl, regions = selectivesearch.selective_search(img, **params)
    n_bbox = len(regions)
    bboxes = np.zeros((n_bbox, 4), dtype=int)
    sizes = np.zeros(n_bbox, dtype=int)
    for i in xrange(n_bbox):
        bboxes[i] = regions[i]['rect']
        sizes[i] = regions[i]['size']
    keep_inds = np.where(sizes > keep_size)[0]
    return bboxes[keep_inds]

if __name__ == '__main__':
    args = parse_args()

    import time

    # Get image filenames to compute selective search
    print 'Getting image filenames and keep ids ...'
    with open(args.img_fname, 'r') as f:
        lines = f.readlines()
        suffix = '.' + args.ext
        img_fnames = [ line.strip() + suffix for line in lines ]
        if args.keep_ids is not None:
            with open(args.keep_ids, 'r') as f:
                lines = f.readlines()
                keep_ids = sorted([ int(line.strip()) for line in lines ])
        else:
            keep_ids = range(len(img_fnames))

    # Compute selective search bboxes
    print 'Computing selective search bboxes ... '
    ssearch_params = {'scale': 500, 'sigma': 0.9, 'min_size': 10}
    t_start = time.time()
    all_boxes = []
    keep_id_ind = 0
    for i,img_fname in enumerate(img_fnames):
        print '{0} / {1}'.format(i+1, len(img_fnames))
        if i == keep_ids[keep_id_ind]:
            boxes = selective_search(img_fname, ssearch_params)
            if (keep_id_ind + 1) != len(keep_ids):
                keep_id_ind += 1
        else:
            boxes = np.zeros((0, 4), dtype=float)
        all_boxes.append(boxes)
    assert (keep_id_ind + 1) == len(keep_ids), 'Unmatch number of bboxes and keep indexes'
    all_boxes = np.array(all_boxes)
    t_end = time.time()

    # Save the output data
    print("Processed {} images in {:.3f} s".format(
        len(keep_ids), t_end - t_start))
    print 'Save output data'
    output_data = {'all_boxes': all_boxes}
    ofname = osp.join(args.opath, 'ss_data.mat')
    scipy.io.savemat(ofname, output_data)
