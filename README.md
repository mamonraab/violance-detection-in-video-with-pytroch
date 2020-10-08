# fire-detection-in-video-with-pytroch

<!-- Copy this Template. -->
<!-- Describe the title of your article by replacing "How-to Template" with the page name you want to publish to. -->
# How to train CNN + LSTM for video Data

## Overview

In these  tutorial  we are going to build a fire detation model based on videos  , since videos is a very important  soruce for rish informations and there is  deffrent kind of appliactions  that  can help  to improve socity life   , I choice fire detaction   since I have published a paper  in 2019  for that topic and I was need to addrress some of  topics that I faced when I build the detactor  

but before we deep dive  i want to declare that  Fire recognition in video is a problem of spatiotemporal features classification once a model can recognize the spatiotemporal features correctly; it can achieve a good result. 
The most common ways in deep-learning approach to capture and learn spatiotemporal features are: -

1.  CNN and LSTM: -  it uses the Convolutional neural network   as a spatial features extractor, then the extracted features feed into LSTM Layer to learn the temporal relation than using any classification layer such as  ANN or any other approach for learning and classification. This approach can benefit from transfer learning by using a pre-trained model in the CNN layer such as vgg19 , resnet  and other pre-trained models to extract the general spatial features. The transfer learning approach  is a very effective method to build a model with high accuracy, especially when there Is limited small data.

2.  Convlutinal neurl network 3d (Conv3D)   several studies show the excellent ability for Conv3d to learn spatiotemporal relation, and it was able to outperform the (CNN and LSTM) approach. Conv3D convolved on four dimensions the time(frame) and height and width and colors channel. It is simple, fast, and more straightforward to train then (CNN and LSTM).

3.  Convlutn-long shortterm memory (Convlstm)   it extends the LSTM model to have a convolutional structure in both input-to-state and state-to-state transitions. ConvLSTM can capture spatiotemporal correlations consistently. 

Since we have small dataset our  best choice is to use a pre-trained CNN with  LSTM  puting that in minde  the  2d CNN can read a 3d input only (C,H,W) and we have a 4d daat which is (frames , c , h , w )  so we need to work around this by doing what kera call it  ( timedistbuted warper )
so the topics of the tutorial as fellows :  
1. bulding custom video data set loader in pytorch 
2.  warping video as a 3d input into  normal conv2d layers this called in keras as ( timedistbuted warper )
3. transffer learning with pytorch

please note that our goal is to keep it simple as ppossible also i didn't like to re-use  same  architcture i used in my aper which gain the stae of the art result in the fire detaction  to leave some  roome to you to improve accuracy and gain better results  the paper  is in this url https://www.researchgate.net/publication/337274826_Robust_Real-Time_Fire_Detector_Using_CNN_And_LSTM

**Keywords:** Optionally add comma-separated keywords.

## Before you start

Make sure you meet the following prerequisites before starting the how-to steps:

* intermidate  pytorch level ( can create class for model , train and evalute modle )
* Prerequisite two have ( pytorch , opencv , pandas ) installed
* Prerequisite three i use 2 datasets   with combntion of some  videos from internet  the 2 datasets links is 
https://zenodo.org/record/836749#.X3zWP3Vficw     https://mivia.unisa.it/datasets/video-analysis-datasets/fire-detection-dataset/   also i downloaded  videos from here  https://pixabay.com/videos/search/fire/


  
## Step-by-step guide

### Step 1: Custom Data loader for videos

handling  video frames data sets wuth an   efficient data generation scheme that consume a less memory can be done  with e Dataset class (torch.utils.data.Dataset) in PyTorch  the idea  is  to privde a class that overriding two subclass functions

 __len__  – returns the size of the dataset

__getitem__  – returns a sample from the dataset given an index.


the main code or the scelton code   for the data is


```
from torch.utils.data import Dataset

class FireDataset(Dataset):
    """Fire dataset."""

    def __init__(self, ----):



    def __len__(self):
        return lenght_of_data

    def __getitem__(self, --):

        sample = {'video': torch.from_numpy(video), 'label': torch.from_numpy(label)}


        return sample
```
now   what we want to do is  we provide a path for the video file  to the abpve  class and make it read the video and  do some transformation and return it with it label  accuroding to this   my class is like the fellwoing :




```
from torch.utils.data import Dataset 

class FireDataset(Dataset):
    """Fire dataset."""

    def __init__(self, datas, timesep=30,rgb=3,h=120,w=120):
        """
        Args:
            datas: pandas dataframe contain path to videos files with label of them
            timesep: number of frames
            rgb: number of color chanles
            h: height
            w: width
                 
        """
        self.dataloctions = datas
        self.timesep,self.rgb,self.h,self.w = timesep,rgb,h,w


    def __len__(self):
        return len(self.dataloctions)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        video = capture(self.dataloctions.iloc[idx, 0],self.timesep,self.rgb,self.h,self.w)
        sample = {'video': torch.from_numpy(video), 'label': torch.from_numpy(np.asarray(self.dataloctions.iloc[idx, 1]))}


        return sample

```

we need now a function called    ( capture )  and a dataframe cotain  path for videos with them label
the  capture funtion is  read the video file and put it i array with same of (timesep,rgb,h,w) also we normalize pixels and if we need to do any transformtion can be done there 
my code for  this function is 

```
def capture(filename,timesep,rgb,h,w):
    tmp = []
    frames = np.zeros((timesep,rgb,h,w), dtype=np.float)
    i=0
    vc = cv2.VideoCapture(filename)
    if vc.isOpened():
        rval , frame = vc.read()
    else:
        rval = False
    frm = resize(frame,(h, w,rgb))
    frm = np.expand_dims(frm,axis=0)
    frm = np.moveaxis(frm, -1, 1)
    if(np.max(frm)>1):
        frm = frm/255.0
    frames[i][:] = frm
    i +=1
    while i < timesep:
        tmp[:] = frm[:]
        rval, frame = vc.read()
        frm = resize(frame,( h, w,rgb))
        frm = np.expand_dims(frm,axis=0)
        if(np.max(frm)>1):
            frm = frm/255.0
        frm = np.moveaxis(frm, -1, 1)
        frames[i-1][:] = frm # - tmp
        i +=1

    return frames

```


<!-- When an image, such as a screenshot, is quicker to interpret than descriptive text, put the screenshot first, otherwise lead with the text. -->

![alt text](https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg "Image title which describes image.")

Brief instructions explaining how to interpret the image.

### Step 2: understand timedistrbution warpper 

as we know  that  most of the pre-trained Conv  based models are Conv2d  where it accept only  a 3d shape  (rgb , height , width )
while video data  each video is a 4d tensor  (frames , rgb , height , width )   ,  also  we know that   we need the pretrained model for spatial feature extraction and will feed it output to a temproal   layer such as (lstm)  
if we ignore the batch size from our calculations than The Time Distribution operation applies the same operation for each group of tensors. The tensor here represents one frame, in the base model, the group of tensors is consist of 30 consecutive frames represented with a shape of (frames , rgb , height , width ) . Each video (a group of tensors) get into the vgg19   as a frame by frame each with the shape of (rgb , height , width ) the vg19  apply same weight same calculation for that group of tensors the calculations changed once new group received. The output of the time distributed vgg19   is a 2d tensor that is feed into the LSTM  

in code the idea is so simple  we just itreat over each frame and feed it  as frame by frame item t the conv base model
from that   i have two  idea to implment this

the first one   ( consume more memory from the gpu  but learn better then the second one)

```
 # reshape input  to be (batch_size * timesteps, input_size)
 x = x.contiguous().view(batch_size * time_steps, C, H, W)
 # feed to the pre-trained conv model
 x = self.baseModel(x)
 # flatten the output
 x = x.view(x.size(0), -1)
 # make the new correct shape (batch_size , timesteps , output_size)
 x = x.contiguous().view(batch_size , time_steps , x.size(-1))  # this x is now ready to be entred or feed into lstm layer
```
 
1. Substep A
1. Substep B
1. Substep C

### Step 3: Optional: title for step - code snippet

Lead-in sentence explaining the code snippet. For example: 

Run the `apt` command to install the Asciidoctor package and check the version.

```
$ sudo apt install asciidoctor

$ asciidoctor --version
Asciidoctor 1.5.6.2 [https://asciidoctor.org]
```

### Step 4: Optional: title for step - Conclusion

Provide a summary of the steps completed and explain what the user has achieved by following them. You can also include links to related articles that may help the reader reinforce concepts discussed in this how-to article.
