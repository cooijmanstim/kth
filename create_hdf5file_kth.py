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
import urllib
import contextlib

def get_split_info():
    def skip_blank_line(file):
        assert not next(file).strip()

    def parse_people_split(line, label):
        match = re.match(r"^\s*%s\s*:\s*person(?P<people>[\d\s,]+)\s*$" % label, line)
        assert match
        people = list(map(int, match.group("people").split(",")))
        return people

    file = urllib.urlopen("http://www.nada.kth.se/cvap/actions/00sequences.txt")

    while next(file).strip() != "following subdivision of sequences with respect to the subject:":
        pass

    skip_blank_line(file)

    people_split = OrderedDict(
        (which_set, parse_people_split(next(file), title))
        for which_set, title in OrderedDict([
                ("train", "Training"),
                ("valid", "Validation"),
                ("test",  "Test")]).items())

    skip_blank_line(file)
    skip_blank_line(file)

    video_infos = OrderedDict(
        (which_set, [])
        for which_set in people_split.keys())

    for line in file:
        if not line.strip():
            continue

        match = re.match(r"^\s*(?P<video_name>person(?P<person>\d+)_"
                         r"(?P<label>[^_]+)_d\d)\s+"
                         r"(?:(?P<missing>[*]missing[*])|"
                         r"frames\s+(?P<ranges>[\d\s,-]+))\s*$",
                         line)
        assert match
        video_info = match.groupdict()
        if video_info["missing"]:
            # wth is that supposed to mean?
            continue
        video_info["person"] = int(video_info["person"])
        video_info["ranges"] = list(list(map(int, ab.split("-")))
                                    for ab in video_info["ranges"].split(","))
        which_set = next(which_set for which_set, people in people_split.items()
                            if video_info["person"] in people)
        video_infos[which_set].append(video_info)
        
    file.close()

    return video_infos

targets_by_label = dict(
    (label, target) for target, label in
    enumerate("boxing handclapping handwaving jogging running walking".split()))

def main():
    if (len(sys.argv) != 2):
        # indir is assumed to have each video's frames layed out like:
        # [cooijmat@bart13 frames]$ find .
        # ./person25_running_d2_uncomp
        # ./person25_running_d2_uncomp/image-1.jpeg
        # ./person25_running_d2_uncomp/image-2.jpeg
        # ./person25_running_d2_uncomp/image-3.jpeg
        # [...]
        # ./person02_handclapping_d4_uncomp
        # ./person02_handclapping_d4_uncomp/image-1.jpeg
        # ./person02_handclapping_d4_uncomp/image-2.jpeg
        # ./person02_handclapping_d4_uncomp/image-3.jpeg
        # [...]
        # and so on.
        print("Usage: %s indir" % sys.argv[0])
        sys.exit(1)

    video_infos_by_set = get_split_info()
    indir = sys.argv[1]

    f = h5py.File(os.environ["KTH_JPEG_HDF5"], "w")

    n_examples = sum(len(video_info["ranges"]) for video_info in
                     itertools.chain.from_iterable(video_infos_by_set.values()))

    # frames are stored in one contiguous table for each subset
    f.create_group("frames")

    # we represent videos by ranges of frames
    f.create_dataset("videos", (n_examples, 2), dtype=np.uint32)
    f.create_dataset("targets", (n_examples,), dtype=np.uint8)

    split_dict = OrderedDict()

    split_stop = 0
    for which_set, video_infos in video_infos_by_set.items():
        frame_paths = []
        videos = []
        targets = []

        for video_info in video_infos:
            print which_set, video_info

            # this one video just doesn't have that many frames
            if video_info["video_name"] == "person01_boxing_d4":
                video_info["ranges"][-1][1] = 304

            for a, b in video_info["ranges"]:
                # the ranges in 00sequences.txt are meant to be inclusive
                b += 1

                # store videos as ranges of frames[which_set]
                videos.append((len(frame_paths), len(frame_paths) + b - a))
                # target is the same for each segment of the video
                targets.append(targets_by_label[video_info["label"]])

                frame_path = os.path.join(
                    indir,
                    video_info["video_name"] + "_uncomp",
                    "image-") + "{}.jpeg"
                frame_paths.extend(map(frame_path.format, xrange(a, b)))

        f["frames"].create_dataset(
            which_set, (len(frame_paths),), maxshape=(None,),
            dtype=h5py.special_dtype(vlen=np.uint8))
        f["frames"][which_set][:] = map(load_frame, frame_paths)

        split_start, split_stop = split_stop, split_stop + len(targets)
        f["videos" ][split_start:split_stop] = videos
        f["targets"][split_start:split_stop] = targets

        split_dict[which_set] = OrderedDict([
            ("videos",  (split_start, split_stop)),
            ("targets", (split_start, split_stop))])

    f.attrs["split"] = H5PYDataset.create_split_array(split_dict)
    f.flush()
    f.close()

def load_frame(path):
    # we store the frames as jpeg and do the decompression on the fly.
    data = StringIO()
    Image.open(path).convert("L").save(data, format="JPEG")
    data.seek(0)
    return np.fromstring(data.read(), dtype="uint8")

if __name__ == "__main__":
    main()
