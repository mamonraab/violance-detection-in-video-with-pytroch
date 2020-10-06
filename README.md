# fire-detection-in-video-with-pytroch

<!-- Copy this Template. -->
<!-- Describe the title of your article by replacing "How-to Template" with the page name you want to publish to. -->
# How to train CNN % LSTM for video Data

## Overview

In these  tutorial  we are going to build a fire detation model based on videos  , since videos is a very important  soruce for rish informations and there is  deffrent kind of appliactions  that  can help  to improve socity life   , I choice fire detaction   since I have published a paper  in 2019  for that topic and I was need to addrress some of  topics that I faced when I build the detactor  

but before we deep dive  i want to declare that  Fire recognition in video is a problem of spatiotemporal features classification once a model can recognize the spatiotemporal features correctly; it can achieve a good result. 
The most common ways in deep-learning approach to capture and learn spatiotemporal features are: -

1. Substep A CNN and LSTM: -  it uses the Convolutional neural network   as a spatial features extractor, then the extracted features feed into LSTM Layer to learn the temporal relation than using any classification layer such as  ANN or any other approach for learning and classification. This approach can benefit from transfer learning by using a pre-trained model in the CNN layer such as vgg19 , resnet  and other pre-trained models to extract the general spatial features. The transfer learning approach  is a very effective method to build a model with high accuracy, especially when there Is limited small data.

1. Substep A Convlutinal neurl network 3d (Conv3D)   several studies show the excellent ability for Conv3d to learn spatiotemporal relation, and it was able to outperform the (CNN and LSTM) approach. Conv3D convolved on four dimensions the time(frame) and height and width and colors channel. It is simple, fast, and more straightforward to train then (CNN and LSTM), the study [12] shows that Conv3D with enough data is the best architecture for action recognition.

1. Substep A Convlutn-long shortterm memory (Convlstm)   it extends the LSTM model to have a convolutional structure in both input-to-state and state-to-state transitions. ConvLSTM can capture spatiotemporal correlations consistently. 

Since we have small dataset our  best choice is to use a pre-trained CNN with  LSTM  puting that in minde  the  2d CNN can read a 3d input only (C,H,W) and we have a 4d daat which is (frames , c , h , w )  so we need to work around this by doing what kera call it  ( timedistbuted warper )
so the topics of the tutorial as fellows :  
1. Substep A bulding custom video data loader in pytorch 
1. Substep A  warping video as a 3d input into  normal conv2d layers this called in keras as ( timedistbuted warper )
1. Substep A transffer learning with pytorch

please note that our goal is to keep it simple as ppossible also i didn't like to re-use  same  architcture i used in my aper which gain the stae of the art result in the fire detaction  to leave some  roome to you to improve accuracy and gain better results  the paper  is in this url https://www.researchgate.net/publication/337274826_Robust_Real-Time_Fire_Detector_Using_CNN_And_LSTM

**Keywords:** Optionally add comma-separated keywords.

## Before you start
<!-- Delete this section if your readers can dive straight into the lesson without requiring any prerequisite knowledge. -->
Make sure you meet the following prerequisites before starting the how-to steps:

* intermidate  pytorch level ( can create class for model , train and evalute modle )
* Prerequisite two have ( pytorch , opencv , pandas ) installed
* Prerequisite three i use 2 datasets   with combntion of some  videos from internet  the 2 datasets links is 
https://zenodo.org/record/836749#.X3zWP3Vficw     https://mivia.unisa.it/datasets/video-analysis-datasets/fire-detection-dataset/   also i downloaded  videos from here  https://pixabay.com/videos/search/fire/


  
## Step-by-step guide

### Step 1: Costume Data loader for videos

<!-- When an image, such as a screenshot, is quicker to interpret than descriptive text, put the screenshot first, otherwise lead with the text. -->

![alt text](https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg "Image title which describes image.")

Brief instructions explaining how to interpret the image.

### Step 2: Optional: title for step - ordered list

Lead-in sentence for an ordered list:

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
