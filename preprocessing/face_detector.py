import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

#for defining abstract base classes (ABCs)
from abc import ABC, abstractmethod
#Ordered dictionaries are just like regular dictionaries but have some extra capabilities relating to ordering operations
from collections import OrderedDict
from typing import List

import cv2
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

from PIL import Image
from facenet_pytorch.models.mtcnn import MTCNN
from torch.utils.data import Dataset


class VideoFaceDetector(ABC):
    
    ######################### Class declaring abstract methods which are defined in FacenetDetector

    def __init__(self, **kwargs) -> None:
        super().__init__()

    @property
    #@abstractmethod - A decorator indicating abstract methods
    @abstractmethod
    def _batch_size(self) -> int:
        pass
    
     #@abstractmethod - A decorator indicating abstract methods
    @abstractmethod
    def _detect_faces(self, frames) -> List:
        pass


class FacenetDetector(VideoFaceDetector):
    
    #To create an MTCNN detector that runs on the GPU, the model is instantiated with device='cuda:0'
    def __init__(self, device="cuda:0") -> None:
        #call the constructeur of VideoFaceDetector class
        super().__init__()
        #create a face detection pipeline using MTCNN without margin to add to bounding box, thresholds set according to dataset and device on which to run neural net passes.)
        #(Multi-Task Cascaded Convolutional Neural Networks is a neural network which detects faces and facial landmarks on images)
        self.detector = MTCNN(margin=0,thresholds=[0.85, 0.95, 0.95], device=device)

    def _detect_faces(self, frames) -> List:
        batch_boxes, *_ = self.detector.detect(frames, landmarks=False)
        return [b.tolist() if b is not None else None for b in batch_boxes]

    @property
    def _batch_size(self):
        return 32


class VideoDataset(Dataset):

    def __init__(self, videos) -> None:
        #call the constructeur of DataSet class
        super().__init__()
        self.videos = videos

    def __getitem__(self, index: int):
        
        
        video = self.videos[index]
        capture = cv2.VideoCapture(video)
        #total number of frame in a file
        frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = OrderedDict()
        for i in range(frames_num):
            #Gets the frame and holds it for further processing
            capture.grab()
            #Decodes and returns the grabbed video frame
            success, frame = capture.retrieve()
            ##Check if we succeeded
            if not success:
                continue
            # cvtColor convert an image from one color space to another -> Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            #resize to a smaller frame
            frame = frame.resize(size=[s // 2 for s in frame.size])
            frames[i] = frame
        return video, list(frames.keys()), list(frames.values())

    def __len__(self) -> int:
        return len(self.videos)
