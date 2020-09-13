import h5py
import numpy as np

path = 'dataset/nyu_depth_v2_labeled.mat'
data = h5py.File(path, 'r')

writeData = np.rot90(np.array(data['images']), axes=(-1,-3))[:, :, ::-1, :]
np.savez_compressed("dataset/image_orig.npz", writeData)
writeData = np.rot90(np.array(data['depths']), axes=(-1,-2))
np.savez_compressed("dataset/image_depth.npz", writeData)
writeData = np.rot90(np.array(data['labels']), axes=(-1,-2))
np.savez_compressed("dataset/image_labels.npz", writeData)
