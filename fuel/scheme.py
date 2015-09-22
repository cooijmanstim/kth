from fuel.schemes import IterationScheme
from picklable_itertools import imap

class HDF5SequentialScheme(IterationScheme):
    # requests slices of the "images" source that correspond to a
    # single video
    requests_examples = True

    def __init__(self, video_ranges, *args, **kwargs):
        self.video_ranges = video_ranges
        super(HDF5SequentialScheme, self).__init__(*args, **kwargs)

    def get_request_iterator(self):
        return imap(slice, self.video_ranges)
