# violance-detection-in-video-with-pytroch

<!-- Copy this Template. -->
<!-- Describe the title of your article by replacing "How-to Template" with the page name you want to publish to. -->
# How to train CNN + LSTM for video Data

## Overview



In this tutorial we are going to build a violence detection model based on videos since videos are a very important source for rich information and there is a different kind of applications that can help to improve society life, I choice violence detection since I have published a paper in 2019 for that topic and I needed to address some of the topics that I faced when I build the detector (most of the online PyTorch based solution for feeding 4d tensor to conv2d   not working well I choose the topic to give you a well-tested solution that you can work with any video data and use the cnn+lstm with easy and efficient way )


but before we deep dive I want to declare that violence recognition in video is a problem of spatiotemporal features classification once a model can recognize the spatiotemporal features correctly; it can achieve a good result. 
The most common ways in the deep-learning approach to capture and learn spatiotemporal features are: -

1.  CNN and LSTM: -  it uses the Convolutional neural network as a spatial features extractor, then the extracted features feed into LSTM Layer to learn the temporal relation than using any classification layer such as  ANN or any other approach for learning and classification. This approach can benefit from transfer learning by using a pre-trained model in the CNN layer such as vgg19, resnet, and other pre-trained models to extract the general spatial features. The transfer learning approach is a very effective method to build a model with high accuracy, especially when there Is limited small data.

2.  Convolutional neural network 3d (Conv3D)   several studies show the excellent ability for Conv3d to learn spatiotemporal relation, and it was able to outperform the (CNN and LSTM) approach. Conv3D convolved on four dimensions the time(frame) and height and width and colors channel. It is simple, fast, and more straightforward to train than (CNN and LSTM).

3.  Convolution-long short-term memory (ConvLstm)   it extends the LSTM model to have a convolutional structure in both input-to-state and state-to-state transitions. ConvLSTM can capture spatiotemporal correlations consistently. 

Since we have a small dataset our  best choice is to use a pre-trained CNN with  LSTM  putting that in mind  the  2d CNN can read a 3d input only (C, H, W) and we have a 4d data which is (frames, C, H, W )  so we need to work around this by doing what Keras call it  ( time-distributed warper )

so what we will learn can be summarized as follows :  
1. bulding custom video data set loader in pytorch
2. understand transfer learning and do it the right way
3. warping video as a 3d input into  normal conv2d layers this called in keras as ( TimeDistributed warper )
4. using LSTM inside   Sequential model in pytorch

please note that our goal is to keep it simple as possible also I didn't like to re-use the same  architecture I used in my paper which gain the state of art result in the violence detection  to leave some  room for you to improve accuracy and gain better results  the paper  is in this URL https://www.researchgate.net/publication/336156932_Robust_Real-Time_Violence_Detection_in_Video_Using_CNN_And_LSTM



## Before you start

Make sure you meet the following prerequisites before starting the how-to steps:

* intermediate  PyTorch level ( can create a class for a model, train, and evaluate model )
* Prerequisite two have ( PyTorch, OpenCV, pandas ) installed
* Prerequisite three I trained my model on  3 datasets  and mixed them to be one large data set  if you want to build or train your own model on the same dataset 
these data sets  are (Movies Fight Detection Dataset   https://academictorrents.com/details/70e0794e2292fc051a13f05ea6f5b6c16f3d3635  )  ,  (Hockey Fight Detection Dataset  https://academictorrents.com/details/38d9ed996a5a75a039b84cf8a137be794e7cee89  )  , and (VIOLENT-FLOWS DATABASE  https://www.openu.ac.il/home/hassner/data/violentflows/ )

  
## Step-by-step guide

Note :- if you want a full video to fellow with please check these link that fully describe these project https://youtu.be/MROlMtZayog

### Step 1: Custom Data loader for videos

handling  video frames data sets with an   efficient data generation scheme that consume less memory can be done  with   Dataset class (torch.utils.data.Dataset) in PyTorch  the idea  is  to provide a class that overriding two subclass functions

   __len__  – returns the size of the dataset

  __getitem__  – returns a sample from the dataset given an index.


the main code or the skeleton code   for the data is


```
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    """Video dataset."""

    def __init__(self, ----):



    def __len__(self):
        return lenght_of_data

    def __getitem__(self, --):

        sample = {'video': torch.from_numpy(video), 'label': torch.from_numpy(label)}


        return sample
```
now   what we want to do is  we provide a path for the video file  to the above  class and make it read the video and  do some transformation and return it with its label actually we have two options the first one is to read videos and store it directly  which will  need more memory and use less processing, while the second option is to give file path and  when training phase start  the loader  will read data as a batch by batch this will use less memory and more processing ( more time in training ) I use option 2 coz   it will cost us more money to get GPU with high memory 

according to this   my class is like the following :




```
from torch.utils.data import Dataset 

class VideoDataset(Dataset):
    """Video dataset."""

    def __init__(self, datas, timesep=30,rgb=3,h=120,w=120):
        """
        Args:
            datas: pandas dataframe contain path to videos files with label of them
            timesep: number of frames
            rgb: number of color channels
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
the code is very simple  we get a pandas data frame  contain ( file path, the label of file )  and also we give the class the shape we want the data to be,   we make a function I call it   ( capture )  it simply read the file from the path  and do normalization and shape it as we want and  extract only the needed (frames)  and return it as NumPy array
and then we return both the video from capture and the label after converting them to tensors. 



my code for ( capture ) function is 

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
    frm = resize(frame,(h, w,rgb))  # resize is   =>    from skimage.transform import resize
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

### Step 2: understand transfer learning and do it the right way 

in deep-learning  we try to fit some objective function and optimize the solution iteratively , you can imagine it as a search for solution , one of the import point in these search for solution algorithm is the start point ( in neural network is the initiated weights ) from these came the main idea of transffer learning ( if we can start from  a good start point using a previously trained wights for simlier task ) and here come 2 different appoaches for these
1.  re-train a previously trained model
2.   keep some layers freezed and not train them and train only few layers 

how you can decide what to do and how much layer to freez (  when we use a Conv layers in the model  the deeper you go in the Conv layers the more specialized feature you will learn for the desired task while the earliest layer learn genral  feature that maybe work in deffrent tasks )

to summary  as in  the Deep Learning  book "https://www.amazon.com/Deep-Learning-Adaptive-Computation-Machine/dp/0262035618/" (Transfer learning and domain adaptation refer to the situation where what has been learned in one setting … is exploited to improve generalization in another setting )

now how you decide which approach you go with here is my golden rules that i work with and give me a great result ( for computer vision where  the pretrained model is a Conv based model )

1.   when you have a very small data and the pre-trained model old task is very similar to your current task
you can freez all  the conv layers and train only your FNN  

2.   when you have a very small data and the pre-trained model old task is different  from your current task
you can freez a half of layers or more  for example about you can go from 60% to 95% of  the conv layers in the pre-trained model and train the rest layers with your FNN  

3.   when you have mid to large data and the pre-trained model old task is different  from your current task
you can freez a few layers  for example about you can go from  0% to 10%  of the conv layers in the pre-trained model and train the rest layers with your FNN  

4.   when you have mid to large data and  the pre-trained model is very similar  to your current task
you can freez about from  10% to 65%  of the conv layers in the pre-trained model and train the rest layers with your FNN  

now we know  the theory lets see   how in code we can play with  pre-trained models with pytorch

to download a    pre-trained model   you can get it from torchvision.models
for example see these code

```
import torchvision.models as models
#get full model cNN + FNN

model = models.densenet169(pretrained=True)

#get only the conv layers from a pre-trained densenet169

model = models.densenet169(pretrained=True).features


```


now to freez layers 
the first importnet part is to know the names of layers or blocks or number of them  from the pre-trained model since  freezing layers or blocks  deppend on the names of them and it different from model to model

here i will show you how to do it with densnet  you simply can print the model layers and check them names and do what ever you want to do with them

```
import torchvision.models as models

#get only the conv layers from a pre-trained densenet169
model = models.densenet169(pretrained=True).features
# densenet169  contain 4 denseblock
#you can simply freez any of these by disable the grad from theme like these freez mean not to train these layers
# by default all layers are trainable
#freezing the first block
for param in model.denseblock1.parameters():
            param.requires_grad = False

```

now what if model not have names or blocks for example vgg19  you must pring the model check the layers numbers and one you know the number of each layer you can loop over the model layers and freez the desired layers

just like these here 


```
import torchvision.models as models

#get only the conv layers from a pre-trained densenet169
baseModel = models.vgg19(pretrained=pretrained).features
i = 0
for child in baseModel.children():
    if i < 28:
        for param in child.parameters():
            param.requires_grad = False
    else:
        for param in child.parameters():
            param.requires_grad = True
    i +=1

```



### Step 3: building the TimeDistributed warper and integrate it into the  PyTorch model 

as we know  that  most of the pre-trained Conv  based models are Conv2d  where it accepts only  a 3d shape  (RGB, height, width )
while video data  each video is a 4d tensor  (frames, RGB, height, width ),  also  we know that   we need the pre-trained model for spatial feature extraction and will feed it output to a temporal   layer such as (LSTM)  
if we ignore the batch size from our calculations then The Time Distribution operation applies the same operation for each group of tensors. The tensor here represents one frame, in the base model, the group of tensors is consist of 30 consecutive frames represented with a shape of (frames, RGB, height, width ). Each video (a group of tensors) gets into the vgg19   as a frame by frame each with the shape of (RGB, height, width ) the vg19 applies the same weight and calculation for that group of tensors the calculations changed once new group received. The output of the time distributed vgg19   is a 2d tensor that is feed into the LSTM  

in code the idea is so simple  we just iterate over each frame and feed it  as frame by frame item t the Conv2d base model
from that, I have two ideas to implement this

1. the first method   ( consume more memory from the GPU  )  the idea is to change the array from (batch, frame, RGB, height, width)  to  (batch*frame, RGB, height, width)   so each frame will  be an item that feeds to the base model
the main idea  can be seen  in this code

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
 now if we want to apply this into a full model class  ( the model is so simple to make it  easy to understand ) the code will be like this
 
```

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        num_classes = 1
        dr_rate= 0.2
        pretrained = True
        rnn_hidden_size = 30
        rnn_num_layers = 2
        #get a pretrained vgg19 model ( taking only the cnn layers and fine tun them)
        baseModel = models.vgg19(pretrained=pretrained).features  
        i = 0
        for child in baseModel.children():
            if i < 28:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True
            i +=1

        num_features = 12800
        self.baseModel = baseModel
        self.dropout= nn.Dropout(dr_rate)
        self.rnn = nn.LSTM(num_features, rnn_hidden_size, rnn_num_layers , batch_first=True)
        self.fc2 = nn.Linear(30, 256)
        self.fc3 = nn.Linear(256, num_classes)
    def forward(self, x):
        batch_size, time_steps, C, H, W = x.size()
        # reshape input  to be (batch_size * timesteps, input_size)
        x = x.contiguous().view(batch_size * time_steps, C, H, W)
        x = self.baseModel(x)
        x = x.view(x.size(0), -1)
        #make output as  ( samples, timesteps, output_size)
        x = x.contiguous().view(batch_size , time_steps , x.size(-1))
        x , (hn, cn) = self.rnn(x)
        x = F.relu(self.fc2(x[:, -1, :])) # get output of the last  lstm not full sequence
        x = self.dropout(x)
        x = self.fc3(x)
        return x 

``` 


2. the second option is to loop over frames   ( less memory  but slower than option one also noted that option one  is  better at learning)
the main idea  can be seen in this code


```
        batch_size, time_steps, C, H, W = x.size() #get shape of the input
        output = []
        #loop over each frame
        for i in range(time_steps):
            #input one frame at a time into the basemodel
            x_t = self.baseModel(x[:, i, :, :, :])
            # Flatten the output
            x_t = x_t.view(x_t.size(0), -1)
            #make a list of tensors for the given smaples 
            output.append(x_t)
        #end loop  
        #make output as  ( samples, timesteps, output_size)
        x = torch.stack(output, dim=0).transpose_(0, 1)   # this x is now ready to be entred or feed into lstm layer
```
 now if we want to apply this into a full model class  ( the model is so simple to make it  easy to understand ) the code will be like this


```

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        num_classes = 1
        dr_rate= 0.2
        pretrained = True
        rnn_hidden_size = 30
        rnn_num_layers = 2
        baseModel = models.vgg19(pretrained=pretrained).features
        i = 0
        for child in baseModel.children():
            if i < 28:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True
            i +=1

        num_features = 12800
        self.baseModel = baseModel
        self.dropout= nn.Dropout(dr_rate)
        self.rnn = nn.LSTM(num_features, rnn_hidden_size, rnn_num_layers , batch_first=True)
        self.fc1 = nn.Linear(30, 256)
        self.fc2 = nn.Linear(256, num_classes)
    def forward(self, x):
        batch_size, time_steps, C, H, W = x.size()
        output = []
        for i in range(time_steps):
            #input one frame at a time into the basemodel
            x_t = self.baseModel(x[:, i, :, :, :])
            # Flatten the output
            x_t = x_t.view(x_t.size(0), -1)
            output.append(x_t)
        #end loop
        #make output as  ( samples, timesteps, output_size)
        x = torch.stack(output, dim=0).transpose_(0, 1)
        output = None # clear var to reduce data  in memory
        x_t = None  # clear var to reduce data  in memory
        x , (hn, cn) = self.rnn(x)
        x = F.relu(self.fc1(x[:, -1, :])) # get output of the last  lstm not full sequence
        x = self.dropout(x)
        x = self.fc2(x)
        return x 
```

well the above is so ugly I want to make a class that  can do the time-distribution  with the two methods 
lets do it

```
class TimeWarp(nn.Module):
    def __init__(self, baseModel, method='sqeeze'):
        super(TimeWarp, self).__init__()
        self.baseModel = baseModel
        self.method = method
 
    def forward(self, x):
        batch_size, time_steps, C, H, W = x.size()
        if self.method == 'loop':
            output = []
            for i in range(time_steps):
                #input one frame at a time into the basemodel
                x_t = self.baseModel(x[:, i, :, :, :])
                # Flatten the output
                x_t = x_t.view(x_t.size(0), -1)
                output.append(x_t)
            #end loop
            #make output as  ( samples, timesteps, output_size)
            x = torch.stack(output, dim=0).transpose_(0, 1)
            output = None # clear var to reduce data  in memory
            x_t = None  # clear var to reduce data  in memory
        else:
            # reshape input  to be (batch_size * timesteps, input_size)
            x = x.contiguous().view(batch_size * time_steps, C, H, W)
            x = self.baseModel(x)
            x = x.view(x.size(0), -1)
            #make output as  ( samples, timesteps, output_size)
            x = x.contiguous().view(batch_size , time_steps , x.size(-1))
        return x

```

so if we want to use our class for example with  nn.Sequential  the code can be like this


```
baseModel = models.vgg19(pretrained=pretrained).features

model = nn.Sequential(TimeWarp(baseModel))
```

now let say we want to build a complete model  using cnn+lstm
but first I want to address that there are users that have problems integrating  LSTM in   nn.Sequential     you can see the question in StackOverflow link
https://stackoverflow.com/questions/44130851/simple-lstm-in-pytorch-with-sequential-module
 
I did answer the   question when I creating this article so anyone can use this code  
the idea is simple  PyTorch   make a big  place for us to do our class and ingrate them 
I make this class to extract the last output from LSTM  and get it back to  the Sequential  see this code


```
class extractlastcell(nn.Module):
    def forward(self,x):
        out , _ = x
        return out[:, -1, :]
```

now the full model code can be like this


```
# Create model

num_classes = 1
dr_rate= 0.2
pretrained = True
rnn_hidden_size = 30
rnn_num_layers = 2
baseModel = models.vgg19(pretrained=pretrained).features
i = 0
for child in baseModel.children():
    if i < 28:
        for param in child.parameters():
            param.requires_grad = False
    else:
        for param in child.parameters():
            param.requires_grad = True
    i +=1

num_features = 12800

# Example of using Sequential
model = nn.Sequential(TimeWarp(baseModel),
                       nn.Dropout(dr_rate),
                      nn.LSTM(num_features, rnn_hidden_size, rnn_num_layers , batch_first=True),
                      extractlastcell(),
                        nn.Linear(30, 256),
                      nn.ReLU(),
                       nn.Dropout(dr_rate),
 nn.Linear(256, num_classes)

        )
```

that's it now all done  ,    
I've created a Rest API   with a trained model  you can find it  at this URL
https://github.com/mamonraab/violance-detection-in-video-with-pytroch/tree/main/flaskapp
the pre-trained wights can be downloaded from this url  https://drive.google.com/file/d/1kk_k8frDLuf3YNLKTjx0MiFhnOBt9NV0/view?usp=sharing
the flask files are (web-fight.py  , mamonfight22.py)   
and i make a client example which send the video file to the desired api point  the client in (client.py
)
you can check it and use it for your  work.

and ofcourse you can implement what you learned  and  train a your own model

thanks for your time
