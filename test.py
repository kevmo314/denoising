import numpy as np
from PIL import Image
from ISR.models import RDN, RRDN
import time

img = Image.open('test-lr.png')
rdn = RDN(arch_params={"C": 4, "D": 3, "G": 64, "G0": 64, "T": 10, "x": 2})
rdn.model.load_weights('./weights/rdn-C4-D3-G64-G064-T10-x2/2021-08-12_2248/rdn-C4-D3-G64-G064-T10-x2_epoch028.hdf5')

img.resize(size=(img.size[0]*2, img.size[1]*2), resample=Image.BICUBIC).save('test-baseline.png', 'png')

start = time.perf_counter()
sr_img = rdn.predict(np.array(img))
print(time.perf_counter() - start)
with Image.fromarray(sr_img) as im:
    im.save('test-output.png', 'png')