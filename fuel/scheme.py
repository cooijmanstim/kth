import numpy as np

from fuel import config
from fuel.schemes import ShuffledScheme
from picklable_itertools import imap
from picklable_itertools.extras import partition_all

"""
    Custom Scheme to bridge between dataset which is a list of frames
    where our processing logic is on the videos
    The __init__ will contain a list of videos
    The get_request_iterator will return a list of frames
    since the transformer requires frames
    **Cannot shuffle the video_indexes list directly!
    video_indexes[i-1] == first frame of ith video
    video_indexes[i] == last frame of ith video
    need to keep relative order.
"""
class HDF5ShuffledScheme(ShuffledScheme) :
    def __init__(self, video_indexes,
                 random_sample=True,
                 f_subsample = 1,
                 r_subsample = False,
                 *args, **kwargs) :
        self.rng = kwargs.pop('rng', None)
        if self.rng is None:
            self.rng = np.random.RandomState(config.default_seed)
        self.sorted_indices = kwargs.pop('sorted_indices', False)
        self.frames_per_video = kwargs.pop('frames_per_video', 10)
        self.random_sample = random_sample

        self.f_subsample = f_subsample
        self.r_subsample = r_subsample

        self.video_indexes = video_indexes
        super(HDF5ShuffledScheme, self).__init__(*args, **kwargs)

    def correct_subsample(self, start, end, fpv, subsample):
        max_subsample = (end - start) / float(fpv)
        return min(np.floor(max_subsample).astype(np.int), subsample)


    def get_start_frame(self, start, end, fpv, subsample):
        if self.random_sample:
            return np.random.randint(start, end - subsample * fpv + 1)

        nb_frame = end - start
        if start + nb_frame // 2 + subsample * fpv < end:
            return start + nb_frame // 2
        return max(start, end - subsample * fpv)

    def get_request_iterator(self) :
        indices = list(self.indices)
        self.rng.shuffle(indices)
        fpv = self.frames_per_video

        if self.r_subsample:
            subsample = np.random.randint(1, self.f_subsample)
        else:
            subsample = self.f_subsample

        frames_array = np.empty([len(indices),fpv])
        #each element of indices is the jth video we want
        for j in xrange(len(indices)):
            i = indices[j]
            if i==0 :
                c_subsample = self.correct_subsample(0, self.video_indexes[i],
                                                     fpv, subsample)
                t = self.get_start_frame(0, self.video_indexes[i],
                                         fpv, c_subsample)
            else :
                c_subsample = self.correct_subsample(self.video_indexes[i-1],
                                                     self.video_indexes[i],
                                                     fpv, subsample)
                t = self.get_start_frame(self.video_indexes[i-1],
                                         self.video_indexes[i],
                                         fpv, c_subsample)
            for k in range(fpv):
                frames_array[j][k] = t + c_subsample * k
        frames_array = frames_array.flatten()

        if self.sorted_indices:
            return imap(sorted, partition_all(self.batch_size*fpv, frames_array))
        else:
            return imap(list, partition_all(self.batch_size*fpv, frames_array))


"""
    Custom Scheme to bridge between dataset which is a list of frames
    where our processing logic is on the videos
    The __init__ will contain a list of videos
    The get_request_iterator will return a list of frames
    since the transformer requires frames

    Return video in seq scheme at a predefined temporal location (f_subsample/nb_subsample)*nb_frame
"""
class HDF5SeqScheme(ShuffledScheme) :
    def __init__(self, video_indexes,
                 random_sample=True,
                 f_subsample = 1,
                 nb_subsample = 25,
                 *args, **kwargs) :
        self.rng = kwargs.pop('rng', None)
        if self.rng is None:
            self.rng = np.random.RandomState(config.default_seed)
        self.sorted_indices = kwargs.pop('sorted_indices', False)
        self.frames_per_video = kwargs.pop('frames_per_video', 10)
        self.random_sample = random_sample

        self.f_subsample = f_subsample
        self.nb_subsample = nb_subsample

        self.video_indexes = video_indexes
        super(HDF5SeqScheme, self).__init__(*args, **kwargs)

        ### Shuffle indices
        self.indices_sh = list(self.indices)
        self.rng.shuffle(self.indices_sh)


    def correct_subsample(self, start, end, fpv, subsample):
        max_subsample = (end - start) / float(fpv)
        return min(np.floor(max_subsample).astype(np.int), subsample)


    def get_start_frame(self, start, end, fpv, subsample):

        start_frame =  (np.floor((end - start) *
                                 (float(self.f_subsample) / self.nb_subsample)))
        return min(start+start_frame, end - subsample * fpv)

    def get_request_iterator(self) :
        indices = self.indices_sh
        fpv = self.frames_per_video
        frames_array = np.empty([len(indices),fpv])
        subsample = 1



        #each element of indices is the jth video we want
        for j in xrange(len(indices)):
            #pick fpv frames randomly
            i = indices[j]
            if i==0 :
                c_subsample = self.correct_subsample(0, self.video_indexes[i], fpv, subsample)
                t = self.get_start_frame(0, self.video_indexes[i], fpv, c_subsample)
            else :
                c_subsample = self.correct_subsample(self.video_indexes[i-1], self.video_indexes[i],
                                                     fpv, subsample)
                t = self.get_start_frame(self.video_indexes[i-1], self.video_indexes[i],
                                         fpv, c_subsample)
            for k in range(fpv):
                frames_array[j][k] = t + c_subsample * k
        frames_array = frames_array.flatten()

        if self.sorted_indices:
            return imap(sorted, partition_all(self.batch_size*fpv, frames_array))
        else:
            return imap(list, partition_all(self.batch_size*fpv, frames_array))

