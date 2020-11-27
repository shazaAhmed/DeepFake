# allow you to write a command line interface for your scripts. https://realpython.com/command-line-interfaces-python-argparse/
import argparse
# provides functions for interacting with the operating system
import os
#contains a variety of things to do with random number generation
import random
# provides a consistent interface to creating and working with additional processes | 
import subprocess

#tto limit the number of threads used to the number of cpus I demand
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
#create partial functions in python by using the partial function
#Partial functions allow one to derive a function with x parameters to a function 
#with fewer parameters and fixed values set for the more limited function.
#Example :
#def func(u,v,w,x):
#    return u*4 + v*3 + w*2 + x

#p = partial(func,5,6,7) we define u = 5, v = 6, w = 7
#print(p(8)) x = 8
from functools import partial
#finds all the pathnames matching a specified pattern according to the rules used by the Unix shell
from glob import glob
#used for parallel execution of a function across multiple input values, distributing the input data across processes (data parallelism)
from multiprocessing.pool import Pool
#get the number of cpu's in the system
from os import cpu_count

#library using which we can develop real-time computer vision applications.
#It mainly focuses on image processing, video capture and analysis including features like face detection and object detection.
import cv2

cv2.ocl.setUseOpenCL(False)

#OpenCV will disable threading optimizations and run all it's functions sequentially
#(to avoid strange crashes related to opencv)
cv2.setNumThreads(0)

# to make your loops show a progress meter
from tqdm import tqdm


def compress_video(video, root_dir):
    """
    This function gets, for a given video, the directory that contains the video, compress them and saves them in the compressed directory
    """
    #splits the path and get the parent directory name
    parent_dir = video.split("/")[-2]
    #Join various path components 
    out_dir = os.path.join(root_dir, "compressed", parent_dir)
    #creates all the intermediate directories if they don't exist
    os.makedirs(out_dir, exist_ok=True)
    #get the name of the video from the path
    video_name = video.split("/")[-1]
    out_path = os.path.join(out_dir, video_name)
    lvl = random.choice([23, 28, 32])
    command = "ffmpeg -i {} -c:v libx264 -crf {} -threads 1 {}".format(video, lvl, out_path)
    try:
        #check_output raises an exception if it receives non-zero exit status
        subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except Exception as e:
        print("Could not process vide", str(e))


if __name__ == '__main__':
    
    #ArgumentParser object will hold all the information necessary to parse the command line into Python data types
    parser = argparse.ArgumentParser(
        description="Extracts jpegs from video")
    parser.add_argument("--root-dir", help="root directory", default="/mnt/sota/datasets/deepfake")
     #parse the standard arguments passed to the script
    args = parser.parse_args()
    videos = [video_path for video_path in glob(os.path.join(args.root_dir, "*/*.mp4"))]
    #Pool allows to  parallelize the execution of a function across multiple input values
    with Pool(processes=cpu_count() - 2) as p:
        #Keeping track with a progress bar
        with tqdm(total=len(videos)) as pbar:
            # imap_unordered: this method chops the iterable into a number of chunks which it submits to the process pool as separate tasks.
            # the ordering of the results from the returned iterator are considered arbitrary
            for v in p.imap_unordered(partial(compress_video, root_dir=args.root_dir), videos):
                pbar.update()
