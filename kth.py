import os.path
import numpy as np
from StringIO import StringIO
import cPickle as pkl
import h5py
import PIL.Image as Image
from picklable_itertools import imap, starmap
import fuel.datasets, fuel.streams, fuel.schemes, fuel.transformers

class JpegHDF5Dataset(fuel.datasets.H5PYDataset):
    def __init__(self, which_set, load_in_memory=True):
        file = h5py.File(os.environ["KTH_JPEG_HDF5"], "r")
        super(JpegHDF5Dataset, self).__init__(file, which_sets=(which_set,),
                                              load_in_memory=load_in_memory)
        # grab only the relevant frames because we don't want to keep
        # them all in memory three times
        a, b = self.get_start_stop(file, "train")["videos"]
        _, video_ranges = super(JpegHDF5Dataset, self).get_data(request=slice(a, b))
        # ensure which_set's frames are stored contiguously; this
        # simplifies indexing and mapping indices.
        assert np.array_equal(video_ranges[:-1, 1], video_ranges[1:, 0])
        self.frames = np.array(file["frames"][video_ranges[0, 0]:video_ranges[-1, 1]])
        self.video_range_base = video_ranges[0, 0]
        if load_in_memory:
            file.close()

    def get_data(self, *args, **kwargs):
        targets, video_ranges = super(JpegHDF5Dataset, self).get_data(*args, **kwargs)
        # account for the fact that self.frames is a subsequence
        video_ranges -= self.video_range_base
        videos = list(map(self.video_from_jpegs, video_ranges))
        return targets, videos

    def video_from_jpegs(self, video_range):
        frames = self.frames[video_range[0]:video_range[1]]
        video = np.array(map(self.load_frame, frames))
        return video

    def load_frame(self, jpeg):
        image = Image.open(StringIO(jpeg.tostring()))
        image = np.array(image).astype(np.float32) / 255.0
        return image

if __name__ == "__main__":
    from fuel import streams, schemes, transformers

    train = JpegHDF5Dataset('train', load_in_memory=True)

    trainstream = streams.DataStream.default_stream(
        dataset=train,
        iteration_scheme=schemes.SequentialScheme(train.num_examples, 10))
    # TODO: deal with disproportionately long videos
    # (e.g. sort batch by length and create multiple batches with like lengths)
    #trainstream = transformers.Padding(
    #    trainstream,
    #    mask_sources=["videos"])

    batch = trainstream.get_epoch_iterator(as_dict=True).next()
    x = batch["videos"]
    y = batch["targets"]
    #mask = batch["videos_mask"]

    import itertools
    import scipy.misc
    def save_video(i, (x, y)):
        scipy.misc.imsave(
            "%i_%i.png" % (i, y),
            np.reshape(x,
                       (x.shape[0]*x.shape[1],
                        x.shape[2])))
    list(itertools.starmap(save_video, enumerate(zip(x, y))))

    import pdb; pdb.set_trace()
