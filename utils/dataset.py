import os
import torch
import cv2
import numpy as np 
from torch.utils.data import Dataset 
from sklearn.model_selection import train_test_split
import time

class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ucf101':
            # folder that contains class labels
            root_dir = 'dataset/UCF-101'
            # Save preprocess data into output_dir
            output_dir = 'dataset/ucf101'

            return root_dir, output_dir
        elif database == 'hmdb51':
            # folder that contains class labels
            root_dir = 'dataset/HMDB-51'
            output_dir = 'dataset/hmdb51'

            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

class VideoDataset(Dataset):
    def __init__(self, dataset="ucf101",split='train', clip_len=64, resize_height=256, resize_width=None, 
                       crop_height=224, crop_width=None ,preprocess=False, transform=None):
        self.root_dir, self.output_dir = Path.db_dir(dataset)
        self.clip_len = clip_len
        self.split = split
        self.transform = transform
        
        self.resize_height = resize_height
        if resize_width is None:
            self.resize_width = resize_height
        else:
            self.resize_width = resize_width

        self.crop_height = crop_height
        if crop_width is None:
            self.crop_width = crop_height
        else:
            self.crop_width = crop_width

        # check original database
        if not os.path.exists(self.root_dir):
            raise RuntimeError('Original dataset not found or corrupted. Please double check!')
        # check if the preprocessed dataset exit or not, if not, start to process the data
        if not self._check_preprocess(dataset) or preprocess:
            print('Preprocessing {} dataset, this will take long, \
                   but it will be done only once.'.format(dataset))
            self._preprocess()

        # Obtain all the filenames of files inside all the class folders
        # Going through each class folder one at a time
        folder = os.path.join(self.output_dir, self.split)
        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)

        assert len(labels) == len(self.fnames)
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))    
        # ipdb.set_trace()
        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        # create dataset labels index file
        if dataset == "ucf101":
            if not os.path.exists('dataset/ucf_labels.txt'):
                with open('dataset/ucf_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')

        elif dataset == 'hmdb51':
            if not os.path.exists('dataset/hmdb_labels.txt'):
                with open('dataset/hmdb_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # Loading and preprocessing.
        # ipdb.set_trace()
        buffer = self._load_frames(self.fnames[index])
        # perform data augmentation 
        if self.split == 'test':
            # get the whole frames when testing
            buffer = self._centercrop(buffer, type='partical')
        else:    
            buffer = self._randomcrop(buffer)
            buffer = self._randomflip(buffer)
        buffer = self._normalize(buffer)
        buffer = self._to_tensor(buffer)
        labels = np.array(self.label_array[index])
            
        return torch.from_numpy(buffer.copy()), torch.from_numpy(labels)

    def _check_preprocess(self, dataset):
        # check path to the output_dir
        if not os.path.exists(self.output_dir):
            return False
        elif not os.path.exists(os.path.join(self.output_dir, 'train')):
            return False
        # check image size in output_dir
        num = 0
        for num, video_class in enumerate(os.listdir(os.path.join(self.output_dir, 'train'))):
            for video in os.listdir(os.path.join(self.output_dir, 'train', video_class)):
                video_name = os.path.join(os.path.join(self.output_dir, 'train', video_class, video),
                             sorted(os.listdir(os.path.join(self.output_dir, 'train', video_class, video)))[0])
                image = cv2.imread(video_name)
                if np.shape(image)[0] != self.resize_height or np.shape(image)[1] != self.resize_width:
                    return False
                else:
                    break
            
        if dataset == "ucf101":
            if num == 100:
                return True
            else:
                return False
        elif dataset == "hmdb51":
            if num == 50:
                return True
            else:
                return False

    def _preprocess(self):
        start_time = time.time()
        # create corresponding folders
        if not os.path.exists(self.output_dir):
            os.makedirs(os.path.join(self.output_dir, 'train'))
            os.makedirs(os.path.join(self.output_dir, 'val'))
            os.makedirs(os.path.join(self.output_dir, 'test'))

        # Split train/val/test sets
        for file in os.listdir(self.root_dir):
            file_path = os.path.join(self.root_dir, file)
            video_files = [name for name in os.listdir(file_path)]
            train_and_valid, test = train_test_split(video_files, test_size=0.2, random_state=42)
            train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)
            # create corresponding video folders in train/val/test sets 
            train_dir = os.path.join(self.output_dir, 'train', file)
            val_dir = os.path.join(self.output_dir, 'val', file)
            test_dir = os.path.join(self.output_dir, 'test', file)
            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            if not os.path.exists(val_dir):
                os.mkdir(val_dir)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)
            # processing video
            for video in train:
                self._process_video(video, file, train_dir)
            for video in val:
                self._process_video(video, file, val_dir)
            for video in test:
                self._process_video(video, file, test_dir)

        print('Preprocessing finished. Spended time is: {}min'.format((time.time() - start_time)/60))

    def _process_video(self, video, action_name, save_dir):
        # Initialize a VideoCapture object to read video data into a numpy array
        video_filename = video.split('.')[0]
        if not os.path.exists(os.path.join(save_dir, video_filename)):
            os.mkdir(os.path.join(save_dir, video_filename))

        # get some attributes of the video
        capture = cv2.VideoCapture(os.path.join(self.root_dir, action_name, video))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Make sure splited video has at least self.clip_len frames
        EXTRACT_FREQUENCY = 4
        if frame_count // EXTRACT_FREQUENCY <= self.clip_len:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= self.clip_len:
                EXTRACT_FREQUENCY -= 1
                if frame_count // EXTRACT_FREQUENCY <= self.clip_len:
                    EXTRACT_FREQUENCY -= 1

        count = 0
        i = 0
        retaining = True
        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if not retaining and frame is None:
                continue
            # save image with specified frequency
            if count % EXTRACT_FREQUENCY == 0:
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)
                i += 1
            count += 1

        # Release the VideoCapture once it is no longer needed
        capture.release()

    def _load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame
        # if the number of the frame is less than self.clip_len, 
        # then argument the buffer with the last frame 
        if frame_count < self.clip_len:
            add = self.clip_len - frame_count
            temp = np.tile(buffer[-1:], (add,1,1,1))
            buffer = np.concatenate((buffer,temp), axis=0)
        return buffer

    def _randomcrop(self, buffer):
        # randomly select time index for temporal jittering
        time_index = (np.random.randint(buffer.shape[0] - self.clip_len)) if (buffer.shape[0] > self.clip_len) else 0

        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - self.crop_height)
        width_index = np.random.randint(buffer.shape[2] - self.crop_width)

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + self.clip_len,
                 height_index:height_index + self.crop_height,
                 width_index:width_index + self.crop_width, :]
        return buffer

    def _centercrop(self, buffer, type='whole'):
        # calculate the start indices in order to crop the video
        height_index = (buffer.shape[1] - self.crop_height) // 2
        width_index = (buffer.shape[2] - self.crop_width) // 2
        # check the type of cropping 
        if type == 'partical':
            time_index = (np.random.randint(buffer.shape[0] - self.clip_len)) if (buffer.shape[0] > self.clip_len) else 0
            buffer = buffer[time_index:time_index + self.clip_len,
                     height_index:height_index + self.crop_height,
                     width_index:width_index + self.crop_width, :]
            return buffer
        else:
            buffer = buffer[:,height_index:height_index + self.crop_height,
                              width_index:width_index + self.crop_width, :]
            return buffer

    def _randomflip(self, buffer):
        # """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""
        # if np.random.random() < 0.5:
        #     for i, frame in enumerate(buffer):
        #         frame = cv2.flip(buffer[i], flipCode=1)
        #         buffer[i] = cv2.flip(frame, flipCode=1)
        # return buffer
        '''filp the video from right to left'''
        if np.random.random() < 0.5:
            buffer = buffer[::-1]
        return buffer

    def _normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame /= 255.0
            # means, stdevs = [],[]
            # for j in range(3):
            #     pixels = frame[...,j].ravel()
            #     means.append(np.mean(pixels))
            #     stdevs.append(np.std(pixels))
            means = [0.485, 0.456, 0.406]
            stdevs = [0.229, 0.224, 0.225]
            frame = (frame - np.array(means)) / np.array(stdevs)
            buffer[i] = frame
        return buffer

    def _to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

if __name__ == "__main__":
    import sys
    import ipdb
    from torch.utils.data import DataLoader
    sys.path.append("..")
    # print(sys.path)
    trainset = VideoDataset(dataset='ucf101', split='train', clip_len=32)
    testset = VideoDataset(dataset='ucf101', split='test', clip_len= 32)
    test_loader = DataLoader(testset, batch_size=4, shuffle=True, num_workers=1)
    for i, (images, labels) in enumerate(test_loader):
        print(images.shape)
        print(labels)
    # buffer = testset._load_frames('/home/birl/ml_dl_projects/bigjun/conv3d/s3d_g_pytorch/dataset/ucf101/train/Archery/v_Archery_g05_c04')
    # buffer = testset._centercrop(buffer)
    # check the number of frames after capturing in each video
    for dataset in (testset.output_dir + x for x in ('/train','/val','/test')):
        total = 0
        for label in sorted(os.listdir(dataset)):
            for video in sorted(os.listdir(os.path.join(dataset,label))):
                frames = os.listdir(os.path.join(dataset,label,video))
                if len(frames) < 64:
                    total+=1
                    print(os.path.join(dataset,label,video), ": ", str(len(frames)))
        print(total)


