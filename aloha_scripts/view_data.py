from argparse import ArgumentParser
from pathlib import Path
import h5py
import numpy as np
import cv2


def decompress_image(image_bytes):
    image = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(image, cv2.IMREAD_COLOR)



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('directory', help='directory containing datafiles')
    args = parser.parse_args()

    directory = Path(args.directory)
    hdf5_files = list(directory.glob('*.hdf5'))

    for file in hdf5_files:
        with h5py.File(file, 'r') as root:
            print(root.attrs['episode_len'], type(root.attrs['episode_len']))
            for timestep in range(root.attrs['episode_len']):
                images = [decompress_image(root[f'/observations/images/{cam_name}/{timestep}.jpg'][()]) for cam_name in root['observations/images'].keys()]
                image = np.concatenate(images, axis=1)
                cv2.imshow('compressed', image)
                cv2.waitKey(1)

