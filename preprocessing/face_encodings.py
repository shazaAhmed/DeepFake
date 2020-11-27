import argparse
import os
from functools import partial
from multiprocessing.pool import Pool

from tqdm import tqdm

from preprocessing.utils import get_original_video_paths

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import random

import face_recognition
import numpy as np


def write_face_encodings(video, root_dir):
    """
    This function gets, for a given video, the directory that contains the cropped images
    the function then creates a random sample of 10 images minimum from this directory 
    and extracts the face encodings from these images and then writes these encodings in a
    file and saves them in the cropped images directory
    """
    video_id, *_ = os.path.splitext(video)
    crops_dir = os.path.join(root_dir, "crops", video_id)
    if not os.path.exists(crops_dir):
        return
    crop_files = [f for f in os.listdir(crops_dir) if f.endswith("jpg")]
    if crop_files:
        crop_files = random.sample(crop_files, min(10, len(crop_files)))
        encodings = []
        for crop_file in crop_files:
            # this part uses the python library face_recognition to create a
            # 128-dimension face encoding for each face in the image.
            img = face_recognition.load_image_file(os.path.join(crops_dir, crop_file))
            encoding = face_recognition.face_encodings(img, num_jitters=10)
            if encoding:
                encodings.append(encoding[0])
        np.save(os.path.join(crops_dir, "encodings"), encodings)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract 10 crops encodings for each video")
    parser.add_argument("--root-dir", help="root directory", default="/home/selim/datasets/deepfake")
    args = parser.parse_args()
    return args


def main():
    """
    This script creates for a given video, a random sample of cropped images. 
    It then uses the python library face_recognition to extract face encodings from each clip and saves these encodings in a file.
    """
    args = parse_args()
    originals = get_original_video_paths(args.root_dir, basename=True)
    with Pool(processes=os.cpu_count() - 4) as p:
        with tqdm(total=len(originals)) as pbar:
            # imap_unordered: this method chops the iterable into a number of chunks which it submits to the process pool as separate tasks.
            # the ordering of the results from the returned iterator are considered arbitrary
            for v in p.imap_unordered(partial(write_face_encodings, root_dir=args.root_dir), originals):
                pbar.update()


if __name__ == '__main__':
    main()
