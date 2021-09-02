"""
Copyright 2021 JaidedAI

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

The MIT License (MIT)

Copyright (c) 2021 NVIDIA CORPORATION

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from PIL import Image
from collections import OrderedDict

import cv2
import numpy as np
from .craft_utils import getDetBoxes, adjustResultCoordinates
from .imgproc import resize_aspect_ratio, normalizeMeanVariance
from .craft import CRAFT
from torch2trt import torch2trt,TRTModule
import os
import time

#for EAST
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from EAST.model import EAST
import os
#from dataset import get_rotate_mat
import numpy as np
import lanms
from EAST.detect import detect

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def test_net(canvas_size, mag_ratio, net, image, text_threshold, link_threshold, low_text, poly, device, estimate_num_chars=False):

    t_det_start=time.time()
    if isinstance(image, np.ndarray) and len(image.shape) == 4:  # image is batch of np arrays
        image_arrs = image
    else:                                                        # image is single numpy array
        image_arrs = [image]

    img_resized_list = []
    # resize - resize_aspect_ratio useless if mag_ratio==1 which it is by default
    for img in image_arrs:
        img_resized, target_ratio, size_heatmap = resize_aspect_ratio(img, canvas_size,
                                                                      interpolation=cv2.INTER_LINEAR,
                                                                      mag_ratio=mag_ratio)
        img_resized_list.append(img_resized)
    ratio_h = ratio_w = 1 / target_ratio #target ratio always 1 if mag_ratio == 1
    # preprocessing
    x = np.array([normalizeMeanVariance(n_img) for n_img in img_resized_list])
    x = Variable(torch.from_numpy(x).permute(0, 3, 1, 2))  # [b,h,w,c] to [b,c,h,w]
    x = x.to(device)

    
    # print("loading trt module")
    # net_trt = TRTModule()
    # net_trt.load_state_dict(torch.load('detector_trt.pth'))
    
    # print("Testing Model Difference:")
    # print(torch.max(torch.abs(net_trt(x)[0] - net(x)[0])))
    # print(torch.max(torch.abs(net_trt(x)[1] - net(x)[1])))

    # forward pass
    with torch.no_grad():
        y, feature = net(x)
        # print("Testing Model Difference in no_grad")
        # y_trt,_ = net_trt(x)
        # print(torch.max(torch.abs(y_trt - y)))
        # y = y_trt


    boxes_list, polys_list = [], []
    for out in y:
        # make score and link map
        score_text = out[:, :, 0].cpu().data.numpy()
        score_link = out[:, :, 1].cpu().data.numpy()

        # Post-processing
        boxes, polys, mapper = getDetBoxes(
            score_text, score_link, text_threshold, link_threshold, low_text, poly, estimate_num_chars)

        # coordinate adjustment
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h) #CRAFT returns original image/2, so this just scales cordinate boxes back up to full size
        polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
        if estimate_num_chars:
            boxes = list(boxes)
            polys = list(polys)
        for k in range(len(polys)):
            if estimate_num_chars:
                boxes[k] = (boxes[k], mapper[k])
            if polys[k] is None:
                polys[k] = boxes[k]
        boxes_list.append(boxes)
        polys_list.append(polys)

    t_det_end=time.time()
#    print("Detection time:",t_det_end-t_det_start)
    return boxes_list, polys_list

def get_detector(trained_model, device='cpu', quantize=True, cudnn_benchmark=False, use_trt=False):
    net = CRAFT()

    if device == 'cpu':
        net.load_state_dict(copyStateDict(torch.load(trained_model, map_location=device)))
        if quantize:
            try:
                torch.quantization.quantize_dynamic(net, dtype=torch.qint8, inplace=True)
            except:
                pass
    else:
        net.load_state_dict(copyStateDict(torch.load(trained_model, map_location=device)))
        net = torch.nn.DataParallel(net).to(device)
        cudnn.benchmark = cudnn_benchmark

    net.eval()
    if use_trt:
        sample_input = torch.randn((1, 3, 480, 640),dtype=torch.float).cuda()
        if os.path.isfile('detector_trt.pth'):
            print("Loading TRT detector")
            net_trt = TRTModule()
            net_trt.load_state_dict(torch.load('detector_trt.pth'))
        else:
            net_trt = torch2trt(net,[sample_input])
            print("Finished converting detector to TRT")
            torch.save(net_trt.state_dict(),'detector_trt.pth')
        print("Testing TRT Model Difference:")
        print(torch.max(torch.abs(net_trt(sample_input)[0] - net(sample_input)[0])))
        print(torch.max(torch.abs(net_trt(sample_input)[1] - net(sample_input)[1])))
        net = net_trt

    return net

def get_textbox(detector, image, canvas_size, mag_ratio, text_threshold, link_threshold, low_text, poly, device, optimal_num_chars=None):
    result = []
    estimate_num_chars = optimal_num_chars is not None
    bboxes_list, polys_list = test_net(canvas_size, mag_ratio, detector,
                                       image, text_threshold,
                                       link_threshold, low_text, poly,
                                       device, estimate_num_chars)
    if estimate_num_chars:
        polys_list = [[p for p, _ in sorted(polys, key=lambda x: abs(optimal_num_chars - x[1]))]
                      for polys in polys_list]

    for polys in polys_list:
        single_img_result = []
        for i, box in enumerate(polys):
            poly = np.array(box).astype(np.int32).reshape((-1))
            single_img_result.append(poly)
        result.append(single_img_result)


    #for EAST


    # model_path  = './EAST/pths/east_vgg16.pth'
    # model = EAST().to(device)
    # model.load_state_dict(torch.load(model_path))
    # model.eval()
    # img_path = './examples/480x640_english.png'
    # img = Image.open(img_path)
    # boxes = detect(img, model, device)
    # print(boxes)
    # result = [[np.array(b,dtype=np.int32) for b in boxes]]
    # print(result)
    # #sample_input = torch.randn((1, 3, 480, 640),dtype=torch.float).cuda()
    # net_trt = torch2trt(model,[img])
    # a = time.time()
    # res = net_trt(img)
    # b = time.time()
    # print("detection time East:",b-a)


    return result

#For EAST Detection
def get_EAST():
    model_path  = './EAST/pths/east_vgg16.pth'
    model = EAST().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def get_textbox_EAST(model,img_path,device):
    #img_path = './examples/480x640_english.png'
    img = Image.open(img_path)
    boxes = detect(img, model, device)
    #print(boxes)
    result = [[np.array(b,dtype=np.int32) for b in boxes]]
    #print(result)
    #sample_input = torch.randn((1, 3, 480, 640),dtype=torch.float).cuda()
    net_trt = torch2trt(model,[img])
    a = time.time()
    res = net_trt(img)
    b = time.time()
    print("detection time East:",b-a)
    return EAST

For EAST detection, follow the instructions on the official EAST repository [here]() and clone it inside the EasyOCR/ directory in this project. Then uncomment the correct lines in easyocr.py detect fucntion. Make sure to call reader.readtext with the image path if you are using EAST.