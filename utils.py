import png
import numpy as np

def chw2hwc(data):
    """Rollaxis of data from channel X height X width (CHW) to HWC layout"""
    assert data.shape[0] <= 4, "Image is not CHW layout"
    return np.rollaxis(data, 0, 3)

def hwc2chw(data):
    """Rollaxis of data from height X width X channel (HWC) to CHW layout"""
    assert data.shape[2] <= 4, "Image is not HWC layout"
    return np.rollaxis(data, 2, 0)

def write_mask(name, width, height, data):
    assert name.lower().endswith(".png")
    f = open(name, 'wb')      # binary mode is important
    w = png.Writer(width, height, greyscale=True, bitdepth=8)
    w.write(f, data)
    f.close()
