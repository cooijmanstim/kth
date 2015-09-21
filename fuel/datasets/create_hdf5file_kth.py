import os.path
import numpy as np
import cPickle as pkl
import h5py
import Image
import sys
import os
import glob, re
import cv2
from StringIO import StringIO


from fuel.datasets.hdf5 import H5PYDataset

"""
Take the 3 jpeg and targets files from the train, subtrain and valid repo
and create one hdf5 file
"""
def create_hdf5(videos, indir) :

    data_path= os.path.join(os.environ['UCF101'], "jpeg_data.hdf5")
    f = h5py.File(data_path,'w')
    f.create_group("video_indexes")
    indexes_data = np.array([])

    for split in ["train", "valid",] :
        f, index_data = fill_hdf5(f, split, videos[split], indir)
        indexes_data = np.append(indexes_data, index_data)

    print indexes_data

    split_dict = {
            'train' : {"images": (0, indexes_data[0])},

            'valid' : {"images": (indexes_data[0], indexes_data[1])},

            # 'subtrain': {"flow_x": (indexes_data[0], indexes_data[1]),
            #              "flow_y": (indexes_data[0], indexes_data[1]),
            #              "targets": (indexes_data[0], indexes_data[1])},

            # 'valid' : {"flow_x": (indexes_data[1], indexes_data[2]),
            #            "flow_y": (indexes_data[1], indexes_data[2]),
            #            "targets": (indexes_data[1] ,indexes_data[2])},


            }

    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
    f.flush()
    f.close()

def sort_by_numbers_in_file_name(list_of_file_names):
    def tryint(s):
        try:
            return int(s)
        except:
            return s
    def alphanum_key(s):
        """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
        """
        return [ tryint(c) for c in re.split('(\-?[0-9]+)', s) ]
    def sort_nicely(l):
        """ Sort the given list in the way that humans expect.
        """
        l.sort(key=alphanum_key)
        return l

    return sort_nicely(list_of_file_names)


def fill_hdf5(f, split_name, videos, indir) :


    video_indexes = f["video_indexes"].create_dataset(split_name,
                                                      (len(videos),),
                                                      dtype="uint32")
    frames_list = []
    flow_y_list = []
    totimg = 0
    for i, filename in enumerate(videos):

        print i, filename, indir+'/', filename

        ### Get all jpeg
        filename = os.path.join(indir+'/',filename)
        frames = glob.glob(filename+'/*.jpeg')
        try:
            assert frames != []
        except Exception, e:
            print "Error with ", filename, ": no images found"
            exit(1)
        frames = sort_by_numbers_in_file_name(frames)[::-1]

        totimg += len(frames)
        video_indexes[i] = totimg
        frames_list.append(frames)



    dt_frames = h5py.special_dtype(vlen=np.uint8)
    try :
        frames_dtset = f["images"]
        prev_set_len = len(frames_dtset)
        frames_dtset.resize(prev_set_len+totimg, axis=0)
        index = prev_set_len-1
    except KeyError :
        frames_dtset = f.create_dataset("images", (totimg,),
                                        dtype=dt_frames,
                                        maxshape=(None,))
        index = 0

    for i in xrange(len(frames_list)) :
        print "Process", i, "/", len(frames_list)
        for j in xrange(len(frames_list[i])) :
            ## frames
            pic = Image.open(frames_list[i][j])
            pic = pic.convert('L')
            data = StringIO()
            pic.save(data, format="JPEG")
            data.seek(0)
            data = data.read()
            img = np.fromstring(data, dtype='uint8')
            frames_dtset[index] = img
            index += 1

    #return hdf5 file, split index for images
    return f, index

def read_list(filename):
    l = []
    with open(filename) as fp:
        for line in fp:
            print line.strip()
            l.append(line.strip())

    return l


if __name__ == "__main__":

    if (len(sys.argv) != 3):
        print("%s: annotdir indir" % sys.argv[0])
        exit(1)


    annotdir = sys.argv[1]
    indir = sys.argv[2]

    videos = {}
    videos['train'] = read_list(annotdir + "/train.txt")
    videos['valid'] = read_list(annotdir + "/valid.txt")

    create_hdf5(videos, indir)
