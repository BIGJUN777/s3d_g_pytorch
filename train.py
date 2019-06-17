from __future__ import print_function
import numpy as np
import argparse
import os
import glob
import time
from tqdm import tqdm
import ipdb

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from utils.dataset import VideoDataset
from model.s3d_g import S3D_G

###########################################################################################
#                                 SET SOME ARGUMENTS                                      #
###########################################################################################
# define a string2boolean type function for argparse
def str2bool(arg):
    arg = arg.lower()
    if arg in ['yes', 'true', '1']:
        return True
    elif arg in ['no', 'false', '0']:
        return False
    else:
        # raise argparse.ArgumentTypeError('Boolean value expected!')
        pass

parser = argparse.ArgumentParser(description="separable 3D CNN for action classification!")

parser.add_argument('--batch_size', type=int, default=2,
                    help='batch size: 10')
parser.add_argument('--clip_len', type=int, default=64,
                    help='set time step: 64') 
parser.add_argument('--drop_prob', type=float, default=0.5,
                    help='dropout parameter: 0.2')
parser.add_argument('--lr', type=float, default=0.003,
                    help='learning rate: 0.001')
parser.add_argument('--gpu', type=str2bool, default='true', 
                    help='chose to use gpu or not: True') 
parser.add_argument('--test', type=str2bool, default='true',
                    help="test the model during traing: True")
#  parser.add_argument('--clip', type=int, default=4,
#                      help='gradient clipping: 4')

parser.add_argument('--dataset', type=str, default='ucf101', choices=['ucf101','hmdb51'],
                    help='location of the dataset: ucf101')
parser.add_argument('--pretrained', type=str, default='',
                    help='location of the pretrained model file for training: None')
parser.add_argument('--log_dir', type=str, default='./log',
                    help='path to save the log data like loss\accuracy... : ./log') 
parser.add_argument('--save_dir', type=str, default='./checkpoints',
                    help='path to save the checkpoints: ./checkpoints')

parser.add_argument('--epoch', type=int, default=300,
                    help='number of epochs to train: 300') 
parser.add_argument('--print_every', type=int, default=10,
                    help='number of steps for printing training and validation loss: 10') 
parser.add_argument('--save_every', type=int, default=50,
                    help='number of steps for saving the model parameters: 50')                      
parser.add_argument('--test_every', type=int, default=50,
                    help='number of steps for testing the model: 50') 
# for dataset processing
parser.add_argument('--resize_height',  type=int, default=256,
                    help='resize the height of frames before processing: 256')
parser.add_argument('--resize_width',  type=int, default=256,
                    help='resize the width of frames before processing: 256') 
parser.add_argument('--crop_height',  type=int, default=224,
                    help='crop the height of frames when processing: 224')
parser.add_argument('--crop_width',  type=int, default=224,
                    help='crop the widht of frames when processing: 224')   

args = parser.parse_args() 

###########################################################################################
#                                     TRAIN/TEST MODEL                                    #
###########################################################################################
def run_model(args):
    # set up model
    if args.dataset == "ucf101":
        num_class = 101
    elif args.dataset == "hmdb51":
        num_class = 51
    # train from scratch if args.pretrained==None
    model = S3D_G(num_class=num_class, drop_prob=args.drop_prob)
    resume_epoch = 0    
    if args.pretrained:
        # initialize model with uploaded checkpoint
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['state_dict'])
        resume_epoch = args.pretrained.split('_')[-2]
    # use GPU if available else revert to CPU
    # ipdb.set_trace()
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    print("Training on {}".format(device))
    model.to(device)

    # prepare dataset
    trainset = VideoDataset(dataset=args.dataset, split='train', clip_len=args.clip_len, 
                            resize_height=args.resize_height, resize_width=args.resize_width, 
                            crop_height=args.crop_height, crop_width=args.crop_width)
    valset = VideoDataset(dataset=args.dataset, split='val', clip_len=args.clip_len,
                          resize_height=args.resize_height, resize_width=args.resize_width, 
                          crop_height=args.crop_height, crop_width=args.crop_width)
    testset = VideoDataset(dataset=args.dataset, split='test', clip_len=args.clip_len,
                           resize_height=args.resize_height, resize_width=args.resize_width, 
                           crop_height=args.crop_height, crop_width=args.crop_width)

    train_dataloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    val_dataloader = DataLoader(valset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_size = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)

    # build optimizer && criterion  
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.NLLLoss()
    # criterion = nn.CrossEntropyLoss()
    # scheduler = optim.lr_scheduler.SetpLR(optimizer, step_size=10, gamma=0.1) #the scheduler divides the lr by 10 every 10 epochs

    # set visualization and create folder to save checkpoints
    writer = SummaryWriter(log_dir=args.log_dir)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    '''------------------------------------------TRAINING------------------------------------------'''
    for epoch in range(resume_epoch, resume_epoch+args.epoch):
        # each epoch has a training and validation step
        for phase in ['train', 'val']:
            start_time = time.time()
            running_loss = 0.0
            running_corrects = 0.0
            # for i,(inputs, labels) in tqdm(enumerate(trainval_loaders[phase])):
            for inputs, labels in tqdm(trainval_loaders[phase]):
                inputs, labels = inputs.to(device), labels.to(device)
                if phase == 'train':
                    model.train()
                    model.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(torch.log(outputs), labels)
                    loss.backward()
                    optimizer.step()
                else:
                    model.eval()
                    # turn off the gradients for vaildation, save memory and computations
                    with torch.no_grad():
                        outputs = model(inputs)
                        loss = criterion(torch.log(outputs), labels)
                # accumulate loss of each batch
                running_loss += loss.item() * inputs.size(0)
                # accumulate accuracy of each batch
                top_probs, top_class = outputs.topk(1, dim=1) # or = torch.max(outputs, 1)   
                running_corrects += torch.sum(top_class == labels.data.view(top_class.shape))
            # calculate the loss and accuracy of each epoch
            epoch_loss = running_loss / trainval_size[phase]
            epoch_acc = running_corrects.double() / trainval_size[phase]
            # log trainval datas, and visualize them in the same graph
            if phase == 'train':
                train_loss = epoch_loss  
                train_acc = epoch_acc
            else:
                writer.add_scalars('trainval_loss_epoch', {'train': train_loss, 'val': epoch_loss}, epoch)
                writer.add_scalars('trainval_acc_epoch', {'train': train_acc, 'val': epoch_acc}, epoch)
            # print data
            if (epoch % args.print_every) == 0:
                end_time = time.time()
                print("[{}] Epoch: {}/{} Loss: {} Acc: {} Execution time: {}".format(\
                        phase, epoch+1, args.epoch, epoch_loss, epoch_acc, (end_time-start_time)))
        # save model
        if epoch % args.save_every == (args.save_every -1):
            checkpoint = {
                  'clip_len': args.clip_len,
                  'resize_height': args.resize_height,
                  'resize_width': args.resize_width,
                  'crop_height': args.crop_height,
                  'crop_width': args.crop_width,  
                  'state_dict': model.state_dict()
            }
            save_name = args.dataset + "_checkpoint_" + str(epoch+1) + '_epoch.pth'
            torch.save(checkpoint, os.path.join(args.save_dir, save_name))

        '''------------------------------------------TESTING------------------------------------------'''
        # test the model or not
        if args.test and (epoch % args.test_every == args.test_every-1):
            print('Finished {} epochs training. Start to test the model!'.format(epoch+1))
            # testing model
            model.eval()
            testing_loss = 0.0
            testing_corrects = 0.0
            for inputs, labels in tqdm(test_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                with torch.no_grad():
                    outputs = model(inputs)
                loss = criterion(torch.log(outputs), labels)
                testing_loss += loss.item() * inputs.shape[0]
                top_probs, top_class = outputs.topk(1, dim=1)
                testing_corrects += torch.sum(top_class == labels.data.view(top_class.shape))
            epoch_loss = testing_loss / test_size
            epoch_acc = testing_corrects.double() / test_size
            # log data
            writer.add_scalar('test_loss_epoch', epoch_loss, epoch+1)
            writer.add_scalar('test_acc_epoch', epoch_acc, epoch+1)
            # print something out
            print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch+1, args.epoch, epoch_loss, epoch_acc))

    writer.close()
    print('Finishing training!')
  
if __name__ == "__main__":
    run_model(args)


