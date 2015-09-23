import os.path
import operator, itertools
import numpy as np
import cPickle as pkl
import h5py
from collections import OrderedDict
from PIL import Image
import sys
import os
import glob, re
from StringIO import StringIO
from fuel.datasets.hdf5 import H5PYDataset

def main():
    if (len(sys.argv) != 3):
        print("%s: annotdir indir" % sys.argv[0])
        sys.exit(1)

    annotdir = sys.argv[1]
    indir = sys.argv[2]

    video_names_by_set = OrderedDict(
        (which_set, 
         read_list(os.path.join(annotdir, "%s.txt" % which_set)))
        for which_set in "train valid test".split())

    split_boundaries = [0] + list(map(len, video_names_by_set.values()))
    split_dict = OrderedDict(
        (which_set, dict(
            videos=(a, b),
            targets=(a, b)))
        for which_set, a, b in zip(video_names_by_set.keys(),
                                   split_boundaries[:-1],
                                   split_boundaries[1:]))

    f = h5py.File(os.environ["KTH_JPEG_HDF5"], "w")

    all_video_names = list(itertools.chain(*video_names_by_set.values()))
    n_videos = len(all_video_names)

    # store all videos' frames in sequence
    all_frames = []
    video_boundaries = [0]
    for i, video_name in enumerate(all_video_names):
        globexpr = os.path.join(indir, video_name, "*.jpeg")
        print i, indir, video_name, globexpr
        frames = glob.glob(globexpr)
        if not frames:
            raise RuntimeError("no frames match %s" % globexpr)
        frames = natural_sort(frames)[::-1]
        all_frames.extend(frames)
        video_boundaries.append(len(all_frames))

    n_frames = len(all_frames)

    f.create_dataset(
        "frames", (n_frames,), maxshape=(None,),
        dtype=h5py.special_dtype(vlen=np.uint8))
    f["frames"][:] = map(load_frame, all_frames)

    # we represent videos by ranges of frames
    f.create_dataset("videos", (n_videos, 2), dtype=np.uint32)
    f["videos"][:, :] = list(zip(video_boundaries[:-1], video_boundaries[1:]))

    f.create_dataset("targets", (n_videos,), dtype=np.uint8)
    f["targets"][:] = map(infer_target, all_video_names)
    
    f.attrs["split"] = H5PYDataset.create_split_array(split_dict)
    f.flush()
    f.close()

def load_frame(path):
    # we store the frames as jpeg and do the decompression on the fly.
    data = StringIO()
    Image.open(path).convert("L").save(data, format="JPEG")
    data.seek(0)
    return np.fromstring(data.read(), dtype="uint8")

labels = dict(
    (label, code) for code, label in
    enumerate("boxing handclapping handwaving jogging running walking".split()))
def infer_target(video_name):
    key = video_name.split("_")[1]
    return labels[key]

def read_list(filename):
    with open(filename) as fp:
        return list(map(operator.methodcaller("strip"), fp))

def natural_sort(xs):
    def tryint(s):
        try:
            return int(s)
        except:
            return s
    def alphanum_key(s):
        """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
        """
        return list(map(tryint, re.split("(\-?[0-9]+)", s)))
    xs.sort(key=alphanum_key)
    return xs

if __name__ == "__main__":
    main()
