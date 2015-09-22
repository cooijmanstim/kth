import numpy as np
import PIL.Image as Image
from StringIO import StringIO
from fuel.transformers import Mapping

class JpegsToVideo(Mapping):
    def __init__(self, data_stream, **kwargs):
        super(JpegsToVideo, self).__init__(
            data_stream, data_stream.produces_examples,
            mapping=self.jpegs_to_video,
            add_sources=None, **kwargs)

    def jpegs_to_video(self, data):
        jpeg_strings, targets = data
        video = np.array(map(self.load_frame, jpeg_strings))
        return video

    def load_frame(self, jpeg_string):
        image = Image.open(StringIO(jpeg_string))
        image = np.array(image).astype(np.float32) / 255.0
        return image
