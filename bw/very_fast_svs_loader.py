from torchvision import datasets, transforms
import pyvips 
import random
import numpy as np

#img = pyvips.Image.new_from_file(str("/data/luberjm/data/small/svs/TCGA-MP-A4TD-01A-03-TS3.678DF757-79E3-4E08-883B-8524F3C7B93F.svs"))

format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

def vips2numpy(vi):
    return np.ndarray(buffer=vi.write_to_memory(),
                      dtype=format_to_dtype[vi.format],
                      shape=[vi.height, vi.width, vi.bands])

patch_size = 512
for x in range(1,200):
    img = pyvips.Image.new_from_file(str("/data/luberjm/data/small/svs/TCGA-MP-A4TD-01A-03-TS3.678DF757-79E3-4E08-883B-8524F3C7B93F.svs"))
    rand_i = random.randint(0,img.width-patch_size)
    rand_j = random.randint(0,img.height-patch_size)
    t = img.crop(rand_i,rand_j,patch_size,patch_size)
    print(x)
    print(t)
    t_np = vips2numpy(t)
    tt_np = transforms.ToTensor()(t_np)
    temp = tt_np[:3,:,:]
    print(temp)
