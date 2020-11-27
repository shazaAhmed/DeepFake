import argparse
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from functools import partial
from glob import glob
from multiprocessing.pool import Pool
from os import cpu_count

import cv2
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
from tqdm import tqdm


def extract_video(video, root_dir):
     """
     This function extract frames(images) from video and saves them in format jpegs with a specific quality
     """
    # VideoCapture - class for video capturing from video files, image sequences 
    capture = cv2.VideoCapture(video)
    # The total number of frame in a file
    frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(frames_num):
        #Gets the frame and holds it for further processing
        capture.grab()
        #Decodes and returns the grabbed video frame
        success, frame = capture.retrieve()
        #Check if we succeeded
        if not success:
            continue
        #Get the filename without the extension from a path
        id = os.path.splitext(os.path.basename(video))[0]
        #Saves frames to a particular folder in format jpegs with a specific quality 
        cv2.imwrite(os.path.join(root_dir, "jpegs", "{}_{}.jpg".format(id, i)), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])



if __name__ == '__main__':
    
    
    
    
    #ArgumentParser object will hold all the information necessary to parse the command line into Python data types
    parser = argparse.ArgumentParser(
        description="Extracts jpegs from video")
    parser.add_argument("--root-dir", help="root directory")
    
    #parse the standard arguments passed to the script
    args = parser.parse_args()
    #makedirs create a directory recursively in the path  
    os.makedirs(os.path.join(args.root_dir, "jpegs"), exist_ok=True)
    videos = [video_path for video_path in glob(os.path.join(args.root_dir, "*/*.mp4"))]
    # Pool allows to  parallelize the execution of a function across multiple input values
    with Pool(processes=cpu_count() - 2) as p:
        #Keeping track with a progress bar
        with tqdm(total=len(videos)) as pbar:
            # imap_unordered: this method chops the iterable into a number of chunks which it submits to the process pool as separate tasks.
            # the ordering of the results from the returned iterator are considered arbitrary
            for v in p.imap_unordered(partial(extract_video, root_dir=args.root_dir), videos):
                pbar.update()
