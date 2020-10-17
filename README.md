# violance-detection-in-video-with-pytroch

<!-- Copy this Template. -->
<!-- Describe the title of your article by replacing "How-to Template" with the page name you want to publish to. -->
# How to train CNN + LSTM for video Data

## Overview

In these  tutorial  we are going to build a violance detation model based on videos  , since videos is a very important  soruce for rish informations and there is  deffrent kind of appliactions  that  can help  to improve socity life   , I choice violance detaction   since I have published a paper  in 2019  for that topic and I was need to addrress some of  topics that I faced when I build the detactor  (most of the online pytorch based  solution for  feeding  4d tensor to conv2d is  empgous and  also not working well  i choice the topic  to give you well tested solution that you can work with any video data and use the cnn+lstm  with easy and efficaint way )

but before we deep dive  i want to declare that  violance recognition in video is a problem of spatiotemporal features classification once a model can recognize the spatiotemporal features correctly; it can achieve a good result. 
The most common ways in deep-learning approach to capture and learn spatiotemporal features are: -

1.  CNN and LSTM: -  it uses the Convolutional neural network   as a spatial features extractor, then the extracted features feed into LSTM Layer to learn the temporal relation than using any classification layer such as  ANN or any other approach for learning and classification. This approach can benefit from transfer learning by using a pre-trained model in the CNN layer such as vgg19 , resnet  and other pre-trained models to extract the general spatial features. The transfer learning approach  is a very effective method to build a model with high accuracy, especially when there Is limited small data.

2.  Convlutinal neurl network 3d (Conv3D)   several studies show the excellent ability for Conv3d to learn spatiotemporal relation, and it was able to outperform the (CNN and LSTM) approach. Conv3D convolved on four dimensions the time(frame) and height and width and colors channel. It is simple, fast, and more straightforward to train then (CNN and LSTM).

3.  Convlutn-long shortterm memory (Convlstm)   it extends the LSTM model to have a convolutional structure in both input-to-state and state-to-state transitions. ConvLSTM can capture spatiotemporal correlations consistently. 

Since we have small dataset our  best choice is to use a pre-trained CNN with  LSTM  puting that in minde  the  2d CNN can read a 3d input only (C,H,W) and we have a 4d daat which is (frames , c , h , w )  so we need to work around this by doing what kera call it  ( timedistbuted warper )
so the topics of the tutorial as fellows :  
1. bulding custom video data set loader in pytorch 
2.  warping video as a 3d input into  normal conv2d layers this called in keras as ( timedistbuted warper )
3. using LSTM inside   Sequential model in pytorch

please note that our goal is to keep it simple as ppossible also i didn't like to re-use  same  architcture i used in my aper which gain the stae of the art result in the violance detaction  to leave some  roome to you to improve accuracy and gain better results  the paper  is in this url https://www.researchgate.net/publication/336156932_Robust_Real-Time_Violence_Detection_in_Video_Using_CNN_And_LSTM

**Keywords:** Optionally add comma-separated keywords.

## Before you start

Make sure you meet the following prerequisites before starting the how-to steps:

* intermidate  pytorch level ( can create class for model , train and evalute modle )
* Prerequisite two have ( pytorch , opencv , pandas ) installed
* Prerequisite three i trained my model on  3 datasets  and mixed them to be one large data set  if you want to build or train your own model on same dataset 
these data sets  are (Movies Fight Detection Dataset   https://academictorrents.com/details/70e0794e2292fc051a13f05ea6f5b6c16f3d3635  )  ,  (Hockey Fight Detection Dataset  https://academictorrents.com/details/38d9ed996a5a75a039b84cf8a137be794e7cee89  )  , and (VIOLENT-FLOWS DATABASE  https://www.openu.ac.il/home/hassner/data/violentflows/ )

  
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
now   what we want to do is  we provide a path for the video file  to the above  class and make it read the video and  do some transformation and return it with it label actualy we have to options the first one is to rad videos and store diractily  which will  need more memory and use less processing , while the 2 option is to give file path and  when training begin the loadeer  will read data as batch by batch this will use less memory and more processing ( more time in training )  i use option 2 coz allways we can wait bet it will cost as more money to get GPU with high memory 

accuroding to this   my class is like the fellwoing :




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
the code is very simple  we get a pandas dataframe  contain ( file path , label of file )  and also we give the class the shape we wan the data to be  ,   we make a function i call it   ( capture )  it simply read the file from the path  and do normaliztion and shape it as we want and  extract only the needed (frames)  and return it as numpy array
and than we return both the video  from capture and the label  after convert them to tensors 
we need now a function called      and a dataframe cotain  path for videos with them label.


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



### Step 2: bulding the timedistrbution warper and integrate it into the  pytroch model

as we know  that  most of the pre-trained Conv  based models are Conv2d  where it accept only  a 3d shape  (rgb , height , width )
while video data  each video is a 4d tensor  (frames , rgb , height , width )   ,  also  we know that   we need the pretrained model for spatial feature extraction and will feed it output to a temproal   layer such as (lstm)  
if we ignore the batch size from our calculations than The Time Distribution operation applies the same operation for each group of tensors. The tensor here represents one frame, in the base model, the group of tensors is consist of 30 consecutive frames represented with a shape of (frames , rgb , height , width ) . Each video (a group of tensors) get into the vgg19   as a frame by frame each with the shape of (rgb , height , width ) the vg19  apply same weight same calculation for that group of tensors the calculations changed once new group received. The output of the time distributed vgg19   is a 2d tensor that is feed into the LSTM  

in code the idea is so simple  we just itreat over each frame and feed it  as frame by frame item t the conv base model
from that   i have two  idea to implment this

1. the first method   ( consume more memory from the gpu  )  the idea is to change the array from (batch,frame,rgb,height ,width)  to  (batch*frame , rgb,height ,width)   so each frame will by an itm that feed to the basemodel
the main idea  can bee seen in this code

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


2. the second options is to loop over frames   ( less memory  but slower than option one also noted that the option one  is  better at learning)
the main idea  can bee seen in this code


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

well the above is so ugly  i want to make class that  can do the time-distrbution  with the two methods 
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

so if we want to use our class  for example with   nn.Sequential  the code can be like this


```
baseModel = models.vgg19(pretrained=pretrained).features

model = nn.Sequential(TimeWarp(baseModel))
```

now let say  we want to build complate model from this   using cnn+lstm
first i found that  there is  a users that have  problems  intgrating  lstm in   nn.Sequential     you can see the qustion in stackoverflow link
https://stackoverflow.com/questions/44130851/simple-lstm-in-pytorch-with-sequential-module
i did awnsered them qustion when i  creating this artical so any one can use these code  
the idea is simple  pytorch   make i big  place for us to do our clas and ingrate them 
i make this class to extract the last output from lstm  in get it back to  the Sequential  see this code


```
class extractlastcell(nn.Module):
    def forward(self,x):
        out , _ = x
        return out[:, -1, :]
```

now the full model code can be lke this


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

thats it now all done   
i've created a flask api   with a trained model  you can find it  at this url
https://github.com/mamonraab/violance-detection-in-video-with-pytroch/tree/main/flaskapp

the flask files are (web-fight.py  , mamonfight22.py)   
and i make a client example which send the video file to the desired api point  the client in (client.py
)
you can check it and use it for your  work.

>>  thanks for your time
