import os.path
import numpy as np
import cPickle as pkl
import h5py

from fuel.datasets.hdf5 import H5PYDataset

class JpegHDF5Dataset(H5PYDataset):
    def __init__(self,
                 split="train",
                 name="jpeg_data.hdf5",
                 signature='UCF101',
                 load_in_memory=True):
        data_path = os.path.join(os.environ[signature], name)
        data_file = h5py.File(data_path,'r')

        self.video_indexes = np.array(data_file["video_indexes"][split])
        self.num_video_examples = len(self.video_indexes)

        super(JpegHDF5Dataset, self).__init__(data_file, which_sets=(split,), load_in_memory=load_in_memory)
        data_file.close()



if __name__ == "__main__":

    from fuel.streams import DataStream
    from LeViRe.fuel.transformers.frames_transformer import JpegHDF5Transformer
    from LeViRe.fuel.transformers.flows_transformer import FlowHDF5Transformer
    from LeViRe.fuel.scheme import HDF5ShuffledScheme


    # train = JpegHDF5Dataset('valid', load_in_memory=True)
    # valid = JpegHDF5Dataset('valid', load_in_memory=True)
    # test = JpegHDF5Dataset('test', load_in_memory=True)

    # trainstream = JpegHDF5Transformer(
    #     input_size=(120, 160),
    #     crop_size=(120, 160),
    #     nchannels=1,
    #     targets=False,
    #     data_stream=DataStream(
    #          dataset=train,
    #          iteration_scheme=HDF5ShuffledScheme(train.video_indexes,
    #                                              train.num_video_examples,
    #                                              128)))
    # validstream = JpegHDF5Transformer(
    #     input_size=(64, 64),
    #     crop_size=(64, 64),
    #     nchannels=1,
    #     data_stream=DataStream(
    #         dataset=train,
    #         iteration_scheme=HDF5ShuffledScheme(valid.video_indexes,
    #                                             valid.num_video_examples,
    #                                             128)))
    # teststream = JpegHDF5Transformer(
    #     input_size=(64, 64),
    #     crop_size=(64, 64),
    #     nchannels=1,
    #     data_stream=DataStream(
    #         dataset=train,
    #         iteration_scheme=HDF5ShuffledScheme(test.video_indexes,
    #                                             test.num_video_examples,
    #                                             128)))


    # epoch = trainstream.get_epoch_iterator()
    # for i, batch in enumerate(epoch):
    #     print i,
    # epoch = validstream.get_epoch_iterator()
    # for i, batch in enumerate(epoch):
    #     print i,
    # epoch = teststream.get_epoch_iterator()
    # for i, batch in enumerate(epoch):
    #     print i,


    train = JpegHDF5Dataset('train',
                            load_in_memory=True)
    #test = JpegHDF5Dataset('test',
    #                       name='jpeg_data_flows.hdf5',
    #                       load_in_memory=True)
    #import pdb; pdb.set_trace()


    trainstream = JpegHDF5Transformer(
        crop_size = (240, 320),
        data_stream=DataStream(
            dataset=train,
            iteration_scheme=HDF5ShuffledScheme(train.video_indexes,
                                                train.num_video_examples,
                                                128)))
    #teststream = FlowHDF5Transformer(
    #    data_stream=DataStream(
    #        dataset=test,
    #        iteration_scheme=HDF5ShuffledScheme(test.video_indexes,
    #                                            test.num_video_examples,
    #                                            128)))

    ### Compute mean here FIXME
    epoch = trainstream.get_epoch_iterator()
    mean = np.zeros((240, 320, 3), dtype=np.float32)
    for i, batch in enumerate(epoch):
        mean += batch[0].sum(axis=(0, 1))
        print mean.sum()

    import pdb; pdb.set_trace()

    mean /= (train.num_video_examples * 10)
    np.save(os.path.join(os.environ['UCF101'], 'mean.npy'), mean)
