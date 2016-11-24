"""A library of frame sampling method"""

import os,sys
import os.path as osp
import numpy as np
from numpy.random import randint
from numpy.random import choice as randchoice

POLICY = ('head', 'middle', 'random')

def get_clips(dir_list):
    """
    Get the clip list from a list of directory name
    Arguments
    dir_list: absolute path of directories where each contains frames of a clip

    Return 
    A list of clips with frames ordered by name
    """
    clips = []
    for d in dir_list:
        frames = [ osp.splitext(fname)[0] for fname in os.listdir(d) if osp.isfile(osp.join(d, fname)) ]
        frames = sorted(frames, key=lambda x: int(x))
        clips.append(frames)
    return clips


def sample_frames(clips, policy, w_size):
    """Sample the frames"""
    assert_msg = 'Parameters number unmatch'
    if policy == 'head':
        return _window_sample(clips, w_size)
    elif policy == 'middle':
        return _window_sample(clips, w_size, rand_start=True)
    elif policy == 'random':
        return _window_sample(clips, w_size, rand_start=True, rand_len=True)


def get_abs_path(dir_list, clips, ext):
    """Get absolute path of frames"""
    assert len(dir_list) == len(clips), "Miss match length of dir_list and clips"
    frame_paths = []
    for i,clip in enumerate(clips):
        for frame in clip:
            frame_path = osp.join(dir_list[i], frame)
            frame_paths.append(frame_path)
    return frame_paths


def _window_sample(clips, w_size=60, rand_start=False, rand_len=False):
    """
    Window the top k frames of the clips
    """
    new_clips = []
    for clip in clips:
        start = 0
        if rand_start and (len(clip) > w_size):
            start = randint(0, len(clip) - w_size)
        if rand_len:
            offset = randint(0, int(w_size / 2))
            flip = randchoice([1,-1])
            w_size = w_size + flip * offset
        new_clip = clip[start:start + w_size]
        new_clips.append(new_clip)
    return new_clips
