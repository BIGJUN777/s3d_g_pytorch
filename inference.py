from __future__ import print_function
import sys
import os
import numpy as np
import random
import cv2
import argparse
import torch
from model.s3d_g import S3D_G
from utils.dataset import VideoDataset
import ipdb

def main(args):
    # set up model
    if args.dataset == "ucf101":
        num_class = 101
    elif args.dataset == "hmdb51":
        num_class = 51
    model = S3D_G(num_class=num_class, gate=args.gate)
    # load checkpoint
    try:
        # initialize model with uploaded checkpoint
        checkpoint = torch.load(args.pretrained, map_location='cuda:0')
        model.load_state_dict(checkpoint['state_dict'])
        # checkpoint = {
        #           'clip_len': 16,
        #           'resize_height': 256,
        #           'resize_width': 256,
        #           'crop_height': 224,
        #           'crop_width': 224,  
        #           'state_dict': model.state_dict()
        # }
        # model.load_state_dict(torch.load(args.pretrained))
        clip_len, resize_height, resize_width, crop_height, crop_width  = checkpoint['clip_len'], checkpoint['resize_height'], \
                                                        checkpoint['resize_width'], checkpoint['crop_height'], checkpoint['crop_width']
        print('Checkpoint loaded!')
    except Exception as e:
        print('Failed to load checkpoint!', e)
        sys.exit(1)
    # use GPU if available else revert to CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.gpu else 'cpu')
    print("Testing on", device)
    model.to(device)
    model.eval()

    # load labels and read video
    with open('./dataset/ucf_labels.txt', 'r') as f:
        class_names = f.readlines()
        f.close()

    # prepare for video
    # ipdb.set_trace()
    if (args.video is not None) and not (args.random_video):
        capture = cv2.VideoCapture(args.video)
    else:
        testset = VideoDataset(dataset=args.dataset, split='test')
        nums = np.arange(len(testset))
        # random.shuffle(nums)
        num = random.choice(nums)
        video = os.path.join(testset.fnames[num].split("/")[0],'UCF-101', \
                             testset.fnames[num].split("/")[3],           \
                             testset.fnames[num].split("/")[4]+'.avi')
        capture = cv2.VideoCapture(video)
        print("Testing on video named {}.avi".format(testset.fnames[num].split("/")[4]))
    retaining = True

    # set up video writer
    fps_video = capture.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    videoWriter = cv2.VideoWriter('result.mp4', fourcc, fps_video, (240, 240))
    # referencing
    clip = []       # vector to keep the frames 
    while retaining:
        if not capture.isOpened():
            capture.open(args.video)
        retaining, frame = capture.read()
        if not retaining and frame is None:
            continue
        temp = cv2.resize(frame, (resize_height, resize_width))
        temp = _center_crop(temp, (crop_height,crop_width))
        temp = _normalize(temp)
        clip.append(temp)
        # the length of frames should be the same as the chip_len when training  
        if len(clip) == clip_len:
            # modify the clip to match the input of model: (bs, channel, temporal_size, height. width)
            inputs  = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0,4,1,2,3))
            inputs = torch.from_numpy(inputs).to(device)
            with torch.no_grad():
                outputs = model(inputs)
            top_probs, top_class = outputs.topk(1, dim=1)
            # show the result on the images
            cv2.putText(frame, class_names[top_class].split(' ')[-1].strip(), (20,20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,225), 1)
            cv2.putText(frame, "prob: %.4f" % top_probs, (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,225), 1)  
            # delete the first frame
            clip.pop(0)
            videoWriter.write(frame)
        cv2.imshow('result', frame)
        cv2.waitKey(10)

    capture.release()
    cv2.destroyAllWindows()         

def _center_crop(frame, crop):
    h, w = (np.array(frame.shape)[:-1] - np.array(crop)) // 2
    frame = frame[h:h+crop[0], w:w+crop[1], :]
    return frame.copy()

def _normalize(frame):
    frame = frame / 255.0
    # means and stds ara the same to those in dataset.py file
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    frame = (frame - np.array(means)) / np.array(stds)
    return frame

def str2bool(arg):
    arg = arg.lower()
    if arg in ['yes', 'true', '1']:
        return True
    elif arg in ['no', 'false', '0']:
        return False
    else:
        # raise argparse.ArgumentTypeError('Boolean value expected!')
        pass

if __name__ == "__main__":
    # set some arguments
    parser = argparse.ArgumentParser(description='inference of the model')
    parser.add_argument('--gate', type=str2bool, default='false', required=True,
                        help='S3D_G(true) or S3D(false): false')
    parser.add_argument('--video', type=str, default='./dataset/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi',
                        help='A path to the test video is necessary.')
    parser.add_argument('--dataset', '-d', type=str, default='ucf101', choices=['ucf101','hmdb51'],
                        help='Location of the dataset: ucf101')
    parser.add_argument('--pretrained', '-p', type=str, default='./checkpoints/ucf101_checkpoint_100_epoch.pth',
                        help='Location of the checkpoint file: ./checkpoints/ucf101_checkpoint_100_epoch.pth')
    parser.add_argument('--gpu', type=str2bool, default='true',
                        help='use GPU or not: true')
    parser.add_argument('--random_video', type=str2bool, default='false',
                        help='select video randomly from the test dataset: false')
    args = parser.parse_args()
    # inferencing
    main(args)