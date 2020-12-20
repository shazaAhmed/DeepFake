# Analysis of the first place solution for the Kaggle competition DeepFake

<!-- Put the link to this slide here so people can follow -->
Complete repo: https://github.com/selimsef/dfdc_deepfake_challenge

---

We will speak about:

- Competition description 
- Data details
- Evaluation description
- Notebook details
- Concept details

---

## Competition description 

- The goal of this Kaggle competition is to develop a deep learning algorithm capable of identifying tricked videos and real videos. 
But what exactly is Deep Fake?
Deepfake is an artificial intelligence technique that consists in generating very realistic synthetic data. It can be applied to different types of data, for example an image, a video, a sound (music, voice), or even writing.
We can thus generate a realistic image from a drawing, colorize images, transfer a style, restore an image, or even give a face expressions, change the gender of the person, and even exchange faces.

---

## Data details - Kaggle 

* **train_sample_videos.zip** - a ZIP file containing a sample set of training videos and a metadata.json with labels. the full set of training videos is available through the links provided above.
* **sample_submission.csv** - a sample submission file in the correct format.
* **test_videos.zip** - a zip file containing a small set of videos to be used as a public validation set.


> the colums are difined as below



| filename | label | original | split  |
| -------- | -------- | -------- | -------- |
|  video's filename| REAL/FAKE     | Original video in case that a train set video is fake     | always equal to train     |


---


## Evaluation description

Submissions are scored on **log loss**:

![](https://i.imgur.com/iDQgB6j.png)

*     n is the number of videos being predicted
*     y^i is the predicted probability of the video being FAKE
*     yi is 1 if the video is FAKE, 0 if REAL
*     log() is the natural (base e) logarithm



# 

> Log Loss is a loss function used in (multinomial) logistic regression based on probabilities. It returns y_pred probabilities for its training data y_true.

> The log loss is only defined for two or more labels. For any given problem, a lower log-loss value means better predictions. 

> Log Loss is a slight twist on something called the Likelihood Function. In fact, Log Loss is -1 * the log of the likelihood function.

> Each prediction is between 0 and 1. If you multiply enough numbers in this range, the result gets so small that computers can't keep track of it. So, as a clever computational trick, we instead keep track of the log of the Likelihood. This is in a range that's easy to keep track of. We multiply this by negative 1 to maintain a common convention that lower loss scores are better.





---
## Notebook details

### Librairies used across all the file 

- **cv2**

> It mainly focuses on image processing, video capture and analysis including features like face detection and object detection. It is used here to read images and create images and it also supports multiple types of image manipulation.

```
img1 = cv2.imread(ori_path, cv2.IMREAD_COLOR)
img2 = cv2.imread(fake_path, cv2.IMREAD_COLOR)
diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
cv2.imwrite(diff_path, diff)
```


- **MTCNN from facenet_pytorch** 

> Multi-Task Cascaded Convolutional Neural Networks is a neural network which detects faces and facial landmarks on images. It's implementation is provided by facenet-pytorch.
```
detector = MTCNN(margin=0, thresholds=[0.65, 0.75, 0.75], device="cpu")
```
- **Image from PIL** 
> PIL is the Python Imaging Library which provides the python interpreter with image editing capabilities. It used for opening, manipulating, and saving many different image file formats. The Image module provides a class with the same name which is used to represent a PIL image.
```
from PIL import Image
pil_img = Image.fromarray(frame_in)
```

- **tqdm** 
> TQDM is a progress bar library with good support for nested loops and Jupyter/IPython notebooks.

> here is a good example of what it does : https://www.geeksforgeeks.org/python-how-to-make-a-terminal-progress-bar-using-tqdm/
```
from tqdm import tqdm

with tqdm(total=len(videos)) as pbar:
    ...........
```


- **defaultdict from collections** 
> Defaultdict is a sub-class of the dict class that returns a dictionary-like object. The functionality of both dictionaries and defualtdict are almost same except for the fact that defualtdict never raises a KeyError . It provides a default value for the key that does not exists.
```
probs = defaultdict(list)
targets = defaultdict(list)
```

- **partial from functools** 
> Partial functions allow one to derive a function with x parameters to a function with fewer parameters and fixed values set for the more limited function.

>Here is a good example of what it does : https://www.learnpython.org/en/Partial_functions#:~:text=You%20can%20create%20partial%20functions,for%20the%20more%20limited%20function.
```
from functools import partial

func = partial(save_diffs, root_dir=args.root_dir)
```


- **Multiprocessing: Pool**
Pool is used to offer a convenient means of parallelizing the execution of a function across multiple input values, distributing the input data across processes (data parallelism)
This is used in order to reduce the processing time and use the full coputational capacity of the machine the code is run on.
```
with Pool(processes=cpu_count() - 2) as p:
    with tqdm(total=len(videos)) as pbar:
        for v in p.imap_unordered(partial(compress_video, root_dir=args.root_dir), videos):
        pbar.update()
```

- **face_recognition**
face_recognition is an easy to use face recognition and manipulation library
```
import face_recognition
img = face_recognition.load_image_file(os.path.join(crops_dir, crop_file))
encoding = face_recognition.face_encodings(img, num_jitters=10)

```

- **albumentations**
A fast image augmentation library and easy to use wrapper around other libraries. It is used heavy augmentations by default
```
from albumentations import ImageCompression, OneOf, GaussianBlur, Blur
img = ImageCompression(quality_lower=40, quality_upper=95)(image=img)["image"]
img = OneOf([GaussianBlur(), Blur()], p=0.5)(image=img)["image"]
```

- **glob** 
> In Python, the glob module is used to retrieve files/pathnames matching a specified pattern. The pattern rules of glob follow standard Unix path expansion rules.
```
for json_path in glob(os.path.join(root_dir, "*/metadata.json")):
```

- **Apex**
> a Pytorch extension with NVIDIA-maintained utilities to streamline mixed precision and distributed training.

> - **apex.amp:** 
> a tool to enable Tensor Core-accelerated training in only 3 lines of Python
```
model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level, loss_scale='dynamic')
```
> - **apex.parallel.DistributedDataParallel:**
> DistributedDataParallel is multi-process and works for both single- and multi- machine training.
```
model = DistributedDataParallel(model, delay_allreduce=True)
```
> - **apex.amp.scale_loss**
> starts the backpropagation
```
amp.scale_loss(loss, optimizer)
```
> - **amp.master_params()**
> generator expression that iterates over the params owned by optimizer
```
amp.master_params(optimizer)
```
- **tensorboardX.SummaryWriter**
>The SummaryWriter class creates an event file in a given directory and add summaries and events to it. The class updates the file contents asynchronously.
```
summary_writer = SummaryWriter(args.logdir + '/' + conf.get("prefix", args.prefix) + conf['encoder'] + "_" + str(args.fold))
```
- **SciPy ndimage**
> The SciPy ndimage submodule is dedicated to image processing.
> - **binarydilation:**
> Dilation is a mathematical morphology operation that uses a structuring element for expanding the shapes in an image.
```
line = binary_dilation(line, iterations=dilation)
```
> - **binary_erosion**
Erosion is a mathematical morphology operation that uses a structuring element for shrinking the shapes in an image.
```
raw_mask = binary_erosion(raw_mask, iterations=random.randint(4, 10))
```

### **Preprocessing**

* #### **Video Compressing**
    To compress videos, FFmpeg is used which is a collection of different projects for handling multimedia files. It specifies libx264 which is an advanced encoding library for creating H.264 (MPEG-4 AVC (a video compression standard )) video streams and uses the rate control mode (CRF) which allows you to keep the best quality and care less about the file size. The range of the CRF scale is 0–51, where 0 is lossless, 23 is the default, and 51 is worst quality possible. The CRF value is randomly chosen between [23, 28, 32]. 
    

* #### **Detect original faces**

    This script detects faces in real videos by using MTCNN detector (which is chosen due to kernel time limits) and store them as jsons. 
    
* #### **Extract crops**

    This script extracts crops in original size and saved them as png. It uses cv2.VideoCapture() to capture a video and bboxes produced by the script detect_original_faces.py to extract useful crops of the videos. These crops are slightly larger than the bboxes that are on the image.
    
* #### **Generate landmarks**

    This script uses MTCCNN (Multi-Task Cascaded Convolutional Neural Networks) to detect faces' bouding boxes and facial landmarks on images. It uses facenet-pytorch's implementation of the MTCCNN.

* #### **Face encodings**
    This script creates for a given video, a random sample of cropped images. It then uses the python library face_recognition to extract face encodings from each clip and saves these encodings in a file.
    
* #### **Generate differences**

    This script takes a pair of an original video and the fake version of it, it computes for a certain number of cropped images (that are in the original and the fake video) the mean structural similarity index between them.
It uses the scikit-image library and in particular the mesure module to compute the SSIM value. It also creats the full SSIM difference image and saves it in a directory.

* #### **Generate Folds**
     This script uses MTCCNN (Multi-Task Cascaded Convolutional Neural Networks) to detect faces’ bouding boxes and facial landmarks on images. It uses facenet-pytorch’s implementation of the MTCCNN.



### **Training**

* #### **Image Compression**
    An application of data compression that encodes the original image with few bits. The objective of image compression is to reduce the redundancy of the image and to store or transmit data in an efficient form.

* #### **Gaussian Noise**

    A statistical noise having a probability density function equal to normal distribution, also known as Gaussian Distribution. Random Gaussian function is added to Image function to generate this noise. It is also called as electronic noise because it arises in amplifiers or detectors.

    It uses a frame-by-frame classification approach.
    MTCNN  for face detection and an EfficientNet.
    B-7 for feature encoding.

* #### **The used encoder: EfficientNet B-7**
    EfficientNet ocuses on improving the accuracy and the efficiency of models.

    Compound Model Scaling

    The paper specifically introduced the method of zooming in a certain dimension alone and the method of compound zooming they proposed.


---

## Concept details : Albumentations / Cutout 

###### **Albumentations - Data Augmentation**

> **Albumentations** is a computer vision tool designed to perform **fast and flexible image augmentations**. It appears to have the largest set of transformation functions of all image augmentation libraries.

![](https://i.imgur.com/uvQVU7J.jpg)

> You can see the whole definition here : [Albumentations](https://github.com/albumentations-team/albumentations#list-of-augmentations)



###### Why is albumentation better?
> The reason this library gained popularity in a small period of time is because of the features it offers. Some of the reasons why this library is better are:
* > **Performance**: Albumentations delivers the best performance on most of the commonly used augmentations. It does this by wrapping several low-level image manipulation libraries and selects the fastest implementation. 
* > **Variety**: This library not only contains the common image manipulation techniques but a wide variety of image transforms. This is helpful for the task and domain-specific applications. 
* > **Flexibility**: Because this package is fairly new, there are multiple image transformations that are proposed and the package has to undergo these changes. But, albumentation has proven to be quite flexible in research and is easily adaptable to the changes.


###### **Cutout - Data Augmentation**

> **Cutout** is an **image augmentation** and regularization technique that randomly masks out square regions of input during training and can be used to improve the robustness and overall performance of convolutional neural networks.


![](https://i.imgur.com/n0bfsCV.png)


> By generating new images which simulate occluded examples, we not only better prepare the model for encounters with occlusions in the real world, but the model also learns to take more of the image context into consideration when making decisions

> In particular, the author uses a class that is called `DeepFakeClassifierDataset` in order to execute his approach to the augmentation of the data set of images.
Augmentation is used to artificially expand the size of a training dataset by creating modified versions of images in the dataset.
The author augments the training and the validation sets by applying one of the following methods on the images:
    > 1. randomly remove one of the landmarks that MTCNN detected on the face randomly: it blacks out landmarks (eyes, nose or mouth)
    > 2. removes/blacks out half face horisontally or vertically and it uses dlib face convex hulls to do that
    > 3. blacks out half the image



---
> Thanks for reading! :100: 