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

    split = OrderedDict(
        (which_set, 
         read_list(os.path.join(annotdir, "%s.txt" % which_set)))
        for which_set in "train valid test".split())

    create_hdf5(split, indir)

def create_hdf5(split, indir):
    f = h5py.File(os.environ["KTH_JPEG_HDF5"], "w")
    f.create_group("video_ranges")

    split_dict = {}
    for which_set, video_names in split.items():
        f, (a, b) = fill_hdf5(f, which_set, video_names, indir)
        split_dict[which_set] = OrderedDict(
            images=(a, b),
            targets=(a, b))

    print split_dict

    f.attrs["split"] = H5PYDataset.create_split_array(split_dict)
    f.flush()
    f.close()

def fill_hdf5(f, which_set, video_names, indir):
    # store all videos' frames in sequence
    all_images = []
    # replicating the target for each video across its frames makes
    # life involving H5PYDataset much easier
    all_targets = []

    video_boundaries = [0]
    for i, video_name in enumerate(video_names):
        globexpr = os.path.join(indir, video_name, "*.jpeg")
        print i, indir, video_name, globexpr
        images = glob.glob(globexpr)
        if not images:
            raise RuntimeError("no images match %s" % globexpr)
        images = natural_sort(images)[::-1]
        all_images.extend(images)
        all_targets.extend(len(images) * [infer_target(video_name)])
        video_boundaries.append(len(all_images))

    n_images = len(all_images)

    # grow hdf5 tables
    for key, dtype in zip("images targets".split(),
                          (h5py.special_dtype(vlen=np.uint8), np.uint8)):
        if key in f:
            f[key].resize(len(f[key]) + n_images, axis=0)
        else:
            f.create_dataset(key, (n_images,),
                             dtype=dtype, maxshape=(None,))

    # insert data
    f["images"][-n_images:] = list(map(load_frame, all_images))
    f["targets"][-n_images:] = all_targets

    # store videos by referring to ranges of images
    video_ranges = f["video_ranges"].create_dataset(
        which_set, (len(video_names), 2), dtype="uint32")
    # note video_ranges are relative to which_set
    video_ranges[:, :] = np.array(zip(video_boundaries[:-1], video_boundaries[1:]))

    # note what range of images and targets corresponds to the subset
    # of the data we just processed
    a, b = len(f["images"]) - n_images, len(f["images"])
    return f, (a, b)

def load_frame(path):
    # we store the images as jpeg and do the decompression on the fly.
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
