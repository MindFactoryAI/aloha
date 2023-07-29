from pathlib import Path
from argparse import ArgumentParser
import os

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('dataset_dir', help='dataset dir to reorder')
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    hdf5_files = list(dataset_dir.glob('*.hdf5'))
    hdf5_files.sort(key=lambda x: os.path.getmtime(x))

    for i, file in enumerate(hdf5_files):
        for ext in ['.hdf5', '_video.mp4', '_qpos.png', '_error.png', '_effort.png', '.jpg']:
            related_file = Path(str(file.parent / file.stem) + ext)
            if related_file.exists():
                os.rename(str(related_file), f'{str(file.parent)}/episode_{i}{ext}')
