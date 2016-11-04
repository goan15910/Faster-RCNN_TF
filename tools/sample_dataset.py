"""Sample frames from original video dataset according to the policy"""

import _init_paths
import os,sys
import os.path as osp
import argparse
from datasets.sample_frames import POLICY, get_clips, sample_frames, get_abs_path

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Sample frames from original dataset')
    parser.add_argument('--root', dest='root_path', help='Absolute root path of the dataset', default=None, type=str)
    parser.add_argument('--policy', dest='policy', help='Sample policy', default='head', type=str)
    parser.add_argument('--window', dest='window', help='Window size', type=int)
    parser.add_argument('--ext', dest='ext', help='Extension of frame files', default="JPEG", type=str)
    parser.add_argument('--opath', dest='opath', help='Output path', default=".", type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def get_leaf_dirs(root_path):
    """Get the list of all leaf directories"""
    leaf_dirs = []
    for d_path, sub_dirs, fnames in os.walk(root_path):
        if not sub_dirs:
            leaf_dirs.append(osp.join(root_path, d_path))
    return leaf_dirs


if __name__ == '__main__':
    args = parse_args()

    print 'Called with args:'
    print args

    policy = args.policy
    w_size = args.window

    if policy in POLICY:
        print 'Sample frames using {:s} policy'.format(policy)
        dir_list = get_leaf_dirs(args.root_path)
        clips = get_clips(dir_list)
        sampled_clips = sample_frames(clips, policy, w_size)
        n_frame = sum(map(len, sampled_clips))
        avg_len = n_frame / len(sampled_clips)
        print 'Average Sample lengths: {:d}'.format(avg_len)
        print 'Total frames number: {:d}'.format(n_frame)
        frame_paths = get_abs_path(dir_list, sampled_clips, args.ext)
    else:
        print POLICY
        print "Invalid policy: ", policy
        sys.exit(1)

    ofname = osp.join(args.opath, "sampled_frames.txt")
    with open(ofname, 'w') as f:
        for frame_path in frame_paths:
            f.write(frame_path + "\n")
