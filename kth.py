import os.path
import numpy as np
from StringIO import StringIO
import cPickle as pkl
import h5py
import PIL.Image as Image
from picklable_itertools import imap, starmap
from fuel.datasets.hdf5 import H5PYDataset
from fuel.transformers import Mapping
from fuel.schemes import IterationScheme

class VideoFramesScheme(IterationScheme):
    # requests slices of the "images" source that correspond to a
    # single video
    requests_examples = True

    def __init__(self, video_ranges, *args, **kwargs):
        self.video_ranges = video_ranges
        super(VideoFramesScheme, self).__init__(*args, **kwargs)

    def get_request_iterator(self):
        return starmap(slice, self.video_ranges)

class JpegsToVideo(Mapping):
    def __init__(self, data_stream, **kwargs):
        super(JpegsToVideo, self).__init__(
            data_stream, mapping=self.jpegs_to_video,
            add_sources=None, **kwargs)

    @property
    def sources(self):
        return ("video", "target")

    def jpegs_to_video(self, data):
        jpegs, targets = data
        video = np.array(map(self.load_frame, jpegs))
        assert np.array_equal(targets[:-1], targets[1:])
        return video, targets[0]

    def load_frame(self, jpeg):
        image = Image.open(StringIO(jpeg.tostring()))
        image = np.array(image).astype(np.float32) / 255.0
        return image

class JpegHDF5Dataset(H5PYDataset):
    def __init__(self, which_set, load_in_memory=True):
        file = h5py.File(os.environ["KTH_JPEG_HDF5"], "r")
        self.video_ranges = np.array(file["video_ranges"][which_set])
        super(JpegHDF5Dataset, self).__init__(file, which_sets=(which_set,),
                                              load_in_memory=load_in_memory)
        if load_in_memory:
            file.close()

    # use get_example_stream or get_batch_stream to get streams,
    # don't use anything else
    def get_example_stream(self):
        stream = streams.DataStream.default_stream(
            dataset=self,
            iteration_scheme=VideoFramesScheme(self.video_ranges))
        stream = JpegsToVideo(stream)
        return stream

    def get_batch_stream(self, batch_size):
        stream = self.get_example_stream()
        stream = transformers.Batch(
            stream,
            schemes.ConstantScheme(
                batch_size=batch_size,
                num_examples=self.num_examples))
        stream = transformers.Rename(
            stream,
            names=dict(video="videos",
                       target="targets"))
        # TODO: deal with disproportionately long videos
        # (e.g. sort batch by length and create multiple batches with like lengths)
        stream = transformers.Padding(
            stream,
            mask_sources=["videos"])
        return stream

if __name__ == "__main__":
    from fuel import streams, schemes, transformers

    train = JpegHDF5Dataset('train', load_in_memory=True)

    trainstream = train.get_batch_stream(100)

    batch = trainstream.get_epoch_iterator(as_dict=True).next()
    x = batch["videos"]
    y = batch["targets"]
    mask = batch["videos_mask"]

    import pdb; pdb.set_trace()
