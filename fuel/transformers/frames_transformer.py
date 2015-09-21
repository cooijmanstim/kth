import numpy as np
import PIL.Image as Image
import time
import cv2

from StringIO import StringIO

from fuel.datasets.hdf5 import H5PYDataset
from fuel.transformers import Transformer


class JpegHDF5Transformer(Transformer) :
    def __init__(self,
                 input_size = (240, 320),
                 crop_size = (224, 224),
                 nchannels = 3,
                 targets=True,
                 flip = True,
                 resize=True,
                 swap_rgb = False,
                 center_crop = False,
                 scale = 1.,
                 nb_frames = 10,
                 mean = None,
                 *args, **kwargs):
        super(JpegHDF5Transformer, self).__init__(*args, **kwargs)

        self.input_size = input_size
        self.crop_size = crop_size
        self.nchannels = nchannels
        self.swap_rgb = swap_rgb
        self.flip = flip
        self.nb_frames = nb_frames
        self.resize = resize
        self.targets = targets
        self.mean = mean
        self.scale = scale
        self.do_center_crop = center_crop
        self.data_sources = ('features', 'targets')

        self.centers = np.array(input_size) / 2.0
        self.centers_crop = (self.centers[0] - self.crop_size[0] / 2.0,
                             self.centers[1] - self.crop_size[1] / 2.0)
        assert self.crop_size[0] <= self.input_size[0]
        assert self.crop_size[1] <= self.input_size[1]
        assert self.nchannels >= 1

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        timer = time.time()

        batch = next(self.child_epoch_iterator)
        images = self.preprocess_data(batch)

        timers2 = time.time()
        print "Returned a minibatch", time.time() - timer, "sec(s)", images[0].shape
        return images

    def preprocess_data(self, batch) :
        #in batch[0] are all the frames. A group of 10 frames is one video
        #in batch[1] are all the targets. They are the same for each 10 elements (1 target for 1 video)
        #frames_per_video=10
        fpv=self.nb_frames

        data_array = batch[0]
        num_videos = int(len(data_array)/fpv)
        x = np.zeros((num_videos, fpv, self.crop_size[0], self.crop_size[1], self.nchannels),
                     dtype='float32')

        if self.targets:
            y = np.empty(num_videos, dtype='int64')

        for i in xrange(num_videos) :
            if self.targets:
                y[i] = batch[1][i*fpv]

            if self.crop_size[0] < self.input_size[0]:
                if self.do_center_crop:
                    x_start = self.centers_crop[0]
                else:
                    x_start = np.random.randint(0, self.input_size[0] - self.crop_size[0])
            else:
                x_start = 0
            if self.crop_size[1] < self.input_size[1]:
                if self.do_center_crop:
                    y_start = self.centers_crop[1]
                else:
                    y_start = np.random.randint(0, self.input_size[1] - self.crop_size[1])
            else:
                y_start = 0
            do_flip = np.random.rand(1)[0]

            for j in xrange(fpv):
                data = data_array[i*fpv+j]
                #this data was stored in uint8
                data = StringIO(data.tostring())
                data.seek(0)
                img = Image.open(data)
                if self.resize and (img.size[0] != 320 and img.size[1] != 240):
                    print "resize", img.size
                    img = img.resize((int(320), int(240)), Image.ANTIALIAS)
                img = (np.array(img).astype(np.float32) / 255.0) * self.scale

                if self.nchannels == 1:
                    img = img[:, :, None]


                if self.mean is not None:
                    img -= self.mean

                if self.swap_rgb and self.nchannels == 3:
                    img = img[:, :, [2, 1, 0]]


                x[i, j, :, :, :] = img[x_start:x_start+self.crop_size[0],
                                       y_start:y_start+self.crop_size[1], :]


                ### Flip
                if self.flip and do_flip > 0.5:
                     new_image = np.empty_like(x[i, j, :, :, :])
                     for c in xrange(self.nchannels):
                         new_image[:,:,c] = np.fliplr(x[i, j, :, :, c])
                     x[i, j, :, :, :] = new_image

                # if self.do_center_crop:
                #     import pdb; pdb.set_trace()
                #     tmp = x[i, j, :, :, :]
                #     cv2.imshow('test', tmpc[:, :, [2, 1, 0]])
                #     cv2.waitKey()



            #cv2.imshow('img', x[i, 0, :, :, :])
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

        self.reshape = False
        if self.reshape == True:
            x = x.reshape((x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4]))
            x = x * 255
            x = x.transpose((0, 3, 1, 2))


        if self.targets:
            return (x, y)
        else:
            return (x,)


