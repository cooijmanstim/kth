import os.path
import numpy as np
import cPickle as pkl
import h5py
from fuel.datasets.hdf5 import H5PYDataset

class JpegHDF5Dataset(H5PYDataset):
    def __init__(self, which_set, load_in_memory=True):
        file = h5py.File(os.environ["KTH_JPEG_HDF5"], "r")
        self.video_ranges = np.array(file["video_ranges"][which_set])
        super(JpegHDF5Dataset, self).__init__(file, which_sets=(which_set,),
                                              load_in_memory=load_in_memory)
        file.close()

if __name__ == "__main__":
    from fuel.streams import DataStream
    from kth.fuel.transformers.frames_transformer import JpegsToVideo
    from kth.fuel.scheme import HDF5SequentialScheme

    train = JpegHDF5Dataset('train',
                            load_in_memory=True)

    trainstream = JpegHDF5Transformer(
        crop_size=(240, 320),
        data_stream=DataStream(
            dataset=train,
            iteration_scheme=HDF5SequentialScheme(
                train.video_ranges,
                128)))

    batch = trainstream.get_epoch_iterator(as_dict=True).next()
    x = batch["features"]
    y = batch["targets"]

    import pdb; pdb.set_trace()
