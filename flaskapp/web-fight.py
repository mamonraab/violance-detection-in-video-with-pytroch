from __future__ import absolute_import
from __future__  import division
from __future__ import print_function
from skimage.io import imread
from skimage.transform import resize
import cv2
import numpy as np
import os
import torch
from mamonfight22 import *
from flask import Flask , request , jsonify
from PIL import Image
from io import BytesIO
import time


np.random.seed(1234)
# Detect devices
gpu = torch.cuda.is_available()
device = torch.device("cuda" if gpu else "cpu")   # use CPU or GPU
model22 = mamon_videoFightModel(device)
model22.to(device)
app = Flask("main-webapi")

@app.route('/api/fight/',methods= ['GET','POST'])
def main_fight(accuracyfight=0.65):
    res_mamon = {}
    if os.path.exists('./tmp.mp4'):
        os.remove('./tmp.mp4')
    filev = request.files['file']
    file = open("tmp.mp4", "wb")
    file.write(filev.read())
    file.close()
    vid = capture("tmp.mp4",40,3,170,170)
    v = np.array([vid])
    v = torch.from_numpy(v)
    v  = v.to(device , dtype=torch.float)
    millis = int(round(time.time() * 1000))
    f , precent = pred_fight(model22,v,acuracy=accuracyfight)
    res_mamon = {'fight':f , 'precentegeoffight':str(precent)}
    millis2 = int(round(time.time() * 1000))
    res_mamon['processing_time'] =  str(millis2-millis)
    resnd = jsonify(res_mamon)
    resnd.status_code = 200
    return resnd

app.run(host='0.0.0.0',port=3091)
