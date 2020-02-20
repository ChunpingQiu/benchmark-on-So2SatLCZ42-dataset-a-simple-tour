# @Date:   2020-02-04T16:58:21+01:00
# @Last modified time: 2020-02-20T20:20:22+01:00

import h5py
import numpy as np

def generator(h5path, batchSize=32, num=None):

    db = h5py.File(h5path, "r")

    indices=np.arange(num)

    while True:

        np.random.shuffle(indices)
        for i in range(0, len(indices), batchSize):

            batch_indices = indices[i:i+batchSize]
            batch_indices.sort()

            by = db["label"][batch_indices,:]
            bx = db["sen2"][batch_indices,:,:,:]

            yield (bx,by)
