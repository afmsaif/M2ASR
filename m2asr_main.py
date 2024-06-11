from numpy.random.mtrand import beta
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random
from ctcdecode import CTCBeamDecoder
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
from conformer import Conformer
import torchaudio
import os
import pandas as pd
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import time
from wer import calculate_wer
###############  hyper-parameters ############

seed = np.random.seed(42)

################    Multihead Conformer Model ########################################

class conformer(nn.Module):
    def __init__(self, num_classes, input_dim, num_encoder_layers, encoder_dim=512):
        super(conformer, self).__init__()
        self.encoder = Conformer(num_classes=1000, 
                  input_dim=input_dim, 
                  encoder_dim=512, 
                  num_encoder_layers=num_encoder_layers)
        
        self.classifier_en = nn.Sequential(
            nn.Linear(1000, 1000),  # birnn returns rnn_dim*2
            # nn.GELU(),
            # nn.Dropout(.01),
            # nn.Linear(num_classes, num_classes)
        )
        self.classifier_zh = nn.Sequential(
            nn.Linear(1000, 5000),  # birnn returns rnn_dim*2
            # nn.GELU(),
            # nn.Dropout(.01),
            # nn.Linear(num_classes, num_classes)
        )
        self.classifier_pre = nn.Sequential(
            nn.Linear(1000, 100),  # birnn returns rnn_dim*2
            # nn.GELU(),
            # nn.Dropout(.01),
            # nn.Linear(num_classes, num_classes)
        )

    def forward(self, x, length):
        x, out_length = self.encoder(x,length)
        # print(x.shape)
        # print(out_length)
        # x = x.view(x.size(0), -1)
        out_en = self.classifier_en(x)
        out_en = F.log_softmax(out_en, dim=-1)
        out_zh = self.classifier_zh(x)
        out_zh = F.log_softmax(out_zh, dim=-1)
        out_pre = self.classifier_pre(x)
        return out_en, out_zh, out_length, out_pre


###########################################################################################

class InfoNCE(nn.Module):

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


############################################################################################

class LibriSpeechDataset(Dataset):
    def __init__(self, audio_files, waveform_length, context_length, future_length, negative_waveform_length):
        self.audio_files = audio_files
        self.waveform_length = waveform_length
        self.context_length = context_length
        self.future_length = future_length
        self.negative_waveform_length = negative_waveform_length

    def __len__(self):
        return len(self.audio_files)

    def load_waveform(self, audio_path, waveform_length):
        waveform, _ = torchaudio.load(audio_path)
        if waveform.size(1) > waveform_length:
            start_idx = random.randint(0, waveform.size(1) - waveform_length)
            waveform = waveform[:, start_idx: start_idx + waveform_length]
        else:
            pad_length = waveform_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        return waveform

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        waveform = self.load_waveform(audio_path, self.waveform_length)

        # Generate context waves
        start_idx = random.randint(0, self.waveform_length - self.context_length - self.future_length)
        context = waveform[:, start_idx: start_idx + self.context_length]

        # Generate future samples
        future = waveform[:, start_idx + self.context_length: start_idx + self.context_length + self.future_length]

        # Generate negative sample
        negative_idx = random.randint(0, len(self.audio_files) - 1)
        while negative_idx == idx:
            negative_idx = random.randint(0, len(self.audio_files) - 1)

        negative_audio_path = self.audio_files[negative_idx]
        negative_waveform = self.load_waveform(negative_audio_path, self.negative_waveform_length)

        negative_sample = negative_waveform

        # Return context, future, negative sample, and waveform length
        return context, future, negative_sample, context.size(1)

###########################################################################################

#########################    projection to simplex  ##########

def projection2simplex(y):
    m = len(y)
    sorted_y = torch.sort(y, descending=True)[0]
    tmpsum = 0.0
    tmax_f = (torch.sum(y) - 1.0)/m
    for i in range(m-1):
        tmpsum+= sorted_y[i]
        tmax = (tmpsum - 1)/ (i+1.0)
        if tmax > sorted_y[i+1]:
            tmax_f = tmax
            break
    return torch.max(y - tmax_f, torch.zeros(m).to(y.device))

def set_seed(seed):
    # torch
    torch.manual_seed(seed)
    # python
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    # cuda
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # dataloaders
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(seed)

    return seed_worker, g


###########################################

def grad_modo(grad_list, **kwargs):
    # get MoCo params
    lambd = kwargs['MoDo']['lambd']
    gamma = kwargs['MoDo']['gamma']
    rho = kwargs['MoDo']['rho']

    # check whether this script use 2 or 3 samples
    double_sample =  True
    if 'num_samples' in kwargs['MoDo']:
        num_samples = kwargs['MoDo']['num_samples']
        if num_samples == 3:
            double_sample =  False

    # grad_list for MoDo contains two gradients
    # print(len(grad_list))
    if double_sample:
        grad1, grad2 = grad_list
    else:
        grad1, grad2, grad3 = grad_list

    # update lambda
    # print('grad1:',grad1.size())
    # print('grad2:',grad2.size())
    # print('lambda:', lambd.size())
    lambd =  projection2simplex( lambd - gamma * ( grad1 @ ( torch.transpose(grad2, 0, 1) @ lambd )  + rho * lambd ) )

    # compute multi-grad
    if double_sample:
        grad_ =  0.5 * lambd @ (grad1 + grad2)
    else:
        grad_ = lambd @ grad3

    # update lambda
    kwargs['MoDo']['lambd'] = lambd

    return grad_, lambd


def find_min_norm_element(grads):

    def _min_norm_element_from2(v1v1, v1v2, v2v2):
        if v1v2 >= v1v1:
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        gamma = -1.0 * ( (v1v2 - v2v2) / (v1v1+v2v2 - 2*v1v2) )
        cost = v2v2 + gamma*(v1v2 - v2v2)
        return gamma, cost

    def _min_norm_2d(grad_mat):
        dmin = 1e8
        for i in range(grad_mat.size()[0]):
            for j in range(i+1, grad_mat.size()[0]):
                c,d = _min_norm_element_from2(grad_mat[i,i], grad_mat[i,j], grad_mat[j,j])
                if d < dmin:
                    dmin = d
                    sol = [(i,j),c,d]
        return sol

    def _next_point(cur_val, grad, n):
        proj_grad = grad - ( torch.sum(grad) / n )
        tm1 = -1.0*cur_val[proj_grad<0]/proj_grad[proj_grad<0]
        tm2 = (1.0 - cur_val[proj_grad>0])/(proj_grad[proj_grad>0])

        skippers = torch.sum(tm1<1e-7) + torch.sum(tm2<1e-7)
        t = torch.ones(1).to(grad.device)
        if (tm1>1e-7).sum() > 0:
            t = torch.min(tm1[tm1>1e-7])
        if (tm2>1e-7).sum() > 0:
            t = torch.min(t, torch.min(tm2[tm2>1e-7]))

        next_point = proj_grad*t + cur_val
        next_point = projection2simplex(next_point)
        return next_point

    MAX_ITER = 250
    STOP_CRIT = 1e-5

    grad_mat = grads.mm(grads.t())
    init_sol = _min_norm_2d(grad_mat)

    n = grads.size()[0]
    sol_vec = torch.zeros(n).to(grads.device)
    sol_vec[init_sol[0][0]] = init_sol[1]
    sol_vec[init_sol[0][1]] = 1 - init_sol[1]

    if n < 3:
        # This is optimal for n=2, so return the solution
        return sol_vec

    iter_count = 0

    while iter_count < MAX_ITER:
        grad_dir = -1.0 * torch.matmul(grad_mat, sol_vec)
        new_point = _next_point(sol_vec, grad_dir, n)

        v1v1 = torch.sum(sol_vec.unsqueeze(1).repeat(1, n)*sol_vec.unsqueeze(0).repeat(n, 1)*grad_mat)
        v1v2 = torch.sum(sol_vec.unsqueeze(1).repeat(1, n)*new_point.unsqueeze(0).repeat(n, 1)*grad_mat)
        v2v2 = torch.sum(new_point.unsqueeze(1).repeat(1, n)*new_point.unsqueeze(0).repeat(n, 1)*grad_mat)

        nc, nd = _min_norm_element_from2(v1v1, v1v2, v2v2)
        new_sol_vec = nc*sol_vec + (1-nc)*new_point
        change = new_sol_vec - sol_vec
        if torch.sum(torch.abs(change)) < STOP_CRIT:
            return sol_vec
        sol_vec = new_sol_vec

    return sol_vec # ADDED

#################################################################

# get layer-wise parameter numbers
def get_layer_params(model):

    # init layer-wise param number list
    num_param_layer = []

    # print layer-wise parameter numbers
    print("\n"+"="*50)
    print('Model parameter count per layer')
    print("="*50)
    # get layerwise param numbers, with layer names
    for name, param in model.named_parameters():
        num_param_layer.append(param.data.numel())
        print(f'{name}', f'\t: {param.data.numel()}')
    print('Total number of parametrs :', sum(num_param_layer))
    print("-"*50)
    # return layerwise and total param numbers
    return sum(num_param_layer), num_param_layer

# get vectorized grad information
def get_grad_vec(model, num_param, num_param_layer):
    # initialize grad with a vecotr with size num. param.
    grad_vec = torch.zeros(num_param)
    # count params to put grad blocks in correct index of vector
    count = 0
    for param in model.parameters():
        # collect grad only if not None, else return zero vector
        if param.grad is not None:
            # calculate vecotr block start and end indices
            beg = 0 if count == 0 else sum(num_param_layer[:count])
            end = sum(num_param_layer[:(count+1)])
            # put flattened grad param into the vector block
            grad_vec[beg:end] = param.grad.data.view(-1)
        count += 1

    return grad_vec

# get gradient and loss values w.r.t each loss function
def get_grads(model, optimizer, num_param, num_param_layer, loss_dict,
                output_e, labels_em, input_lengths_e1, label_lengths_em,
                output_c, labels_cm, input_lengths_c1, label_lengths_cm):
    # init gradient list (to be collected one gradient for each loss)
    grad_list = []
    loss_list = []
    # to switch off retain_graph in loss.backward()
    num_loss = len(loss_dict)
    # print(num_loss)
    # compute the loss w.r.t each loss function
    for k, loss_fn in enumerate(loss_dict):

        if k == 0:
            loss =  loss_dict[loss_fn](output_e, labels_em, input_lengths_e1, label_lengths_em)
        
        elif k==1:
            loss =  loss_dict[loss_fn](output_c, labels_cm, input_lengths_c1, label_lengths_cm)

        # make gradient of model zero
        optimizer.zero_grad()
        # compute loss w.r.t current loss function
        loss.backward(retain_graph=True) if k < num_loss - 1 else loss.backward()
        # compute vectorized gradient
        grad_vec = get_grad_vec(model, num_param, num_param_layer) # [grad1 grad2]
        # collect the gradient for current loss
        grad_list.append(grad_vec)
        loss_list.append(loss.detach().item())

    return torch.stack(grad_list), np.array(loss_list)

# set multi-gradient in the model param grad
def set_grads(model, multi_grad, num_param_layer, device):
    # count params to put multi-grad blocks in correct model param grad
    count = 0
    for param in model.parameters():
        # put grad only if grad is initialized
        if param.grad is not None:
            # calculate vector block start and end indices
            beg = 0 if count == 0 else sum(num_param_layer[:count])
            end = sum(num_param_layer[:(count+1)])
            # put reshaped multi-grad block into model param grad
            param.grad.data = multi_grad[beg:end].view(param.data.size()).data.clone().to(device)
        count += 1
    return


seed_worker, g = set_seed(42)

######################################




""" Load and preprocess data.
"""

class ASR(Dataset):
    """
    Stores a Pandas DataFrame in __init__, and reads and preprocesses examples in __getitem__.
    """
    def __init__(self, split, augmentation):
        """
        Args:
            augmentation (bool): Apply SpecAugment to training data or not.
        """
        if split.upper()=='TRAIN':
            file_path = '/media/chenlab2/hdd5/saif/asr/conformer/TRAIN.csv'
            self.df1 = pd.read_csv(file_path)
            # Load the second dataset
            self.df2 = pd.read_csv(file_path)
            self.df3 = pd.read_csv(file_path)
            self.df4 = pd.read_csv(file_path)
            # Concatenate the datasets
            self.df = pd.concat([self.df1, self.df2, self.df3, self.df4], ignore_index=True)


        # self.df = pd.read_csv('%s.csv' % split.upper())
        # self.tokenizer = torch.load('tokenizer.pth')
            
        if split.upper()=='TEST':
            self.df = pd.read_csv('/media/chenlab2/hdd5/saif/asr/conformer/TEST.csv')
        self.augmentation = (augmentation and (split.upper() == 'TRAIN'))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
            x (torch.FloatTensor, [seq_length, dim_features]): The FBANK features.
            y (torch.LongTensor, [n_tokens]): The label sequence.
        """
        x, y = self.df.iloc[idx]
        x, sample_rate = torchaudio.load(x)


        return x, y


class SentencePieceTransform:
    """Maps subwords to integers and vice versa using SentencePiece"""
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

    def text_to_int(self, text):
        """ Use the SentencePiece tokenizer to convert text to an integer sequence """
        subwords = self.sp.EncodeAsPieces(text.lower())
        return [self.sp.PieceToId(subword) for subword in subwords]

    def int_to_text(self, labels):
        """ Use the SentencePiece tokenizer to convert integer labels to a text sequence """
#         for label in labels:
#             print(str(label))
#         subwords = [self.sp.IdToPiece(label) for label in labels]
#         subwords = [self.sp.decode([int(label)]) for label in labels]
#         return ' '.join(subwords).replace('▁', ' ').strip()
        return self.sp.decode(labels)

sentencepiece_transform_e = SentencePieceTransform("libri1000u.model")
sentencepiece_transform_c = SentencePieceTransform("aishell_unigram5000_model.model")


train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80),
    #torchaudio.transforms.TimeStretch(.8, fixed_rate=True),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=25),
    torchaudio.transforms.TimeMasking(time_mask_param=100),
)


train_audio_transforms_c = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80),
    #torchaudio.transforms.TimeStretch(.8, fixed_rate=True),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=25),
    # torchaudio.transforms.TimeMasking(time_mask_param=torch.randint(50, 100, (1,))),
    # torchaudio.transforms.FrequencyMasking(freq_mask_param=torch.randint(50, 100, (1,))),
    torchaudio.transforms.TimeMasking(time_mask_param=100),
)


valid_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80),
#     torchaudio.transforms.SlidingWindowCmn(cmn_window=500, center=True, norm_vars=False)

)

def data_processing(data, data_type="train"):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []

    for (waveform, _, utterance, _, _, _) in data:

            if data_type == 'train':
                spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
                

            elif data_type == 'test' or "valid":
                spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
                
            else:
                raise Exception('data_type should be train or valid')
            spectrograms.append(spec)
            label = torch.Tensor(sentencepiece_transform_e.text_to_int(utterance))
            labels.append(label)
            input_lengths.append(spec.shape[0])
            label_lengths.append(len(label))


    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    
#     #print(spectrograms.size())
# #    
# #    print(labels.size())

    return spectrograms, labels, input_lengths, label_lengths



def numtoword(beam_results, out_lens, labels, label_lengths, lang = 1, blank_label=0, collapse_repeated=True):
    arg_maxes = beam_results

    decodes = []
    targets = []

    for i, args in enumerate(arg_maxes):
        decode = []
        tar_list = labels[i][:label_lengths[i]].tolist()
        tar_list = list(map(int, tar_list))
        tar_list = list(filter(lambda x: x != 0, tar_list))

        if lang==1:
            targets.append(sentencepiece_transform_e.int_to_text(tar_list))
        elif lang==2:
            targets.append(sentencepiece_transform_c.int_to_text(tar_list))
    
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        if lang==1:
            decodes.append(sentencepiece_transform_e.int_to_text(decode))
        elif lang==2:
            decodes.append(sentencepiece_transform_c.int_to_text(decode))
    return decodes, targets


def data_processing_c(data, data_type="train"):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (waveform, utterance) in data:
        if data_type == 'train':
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
            

        elif data_type == 'test' or "valid":
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
            
        else:
            raise Exception('data_type should be train or valid')
        spectrograms.append(spec)
        label = torch.Tensor(sentencepiece_transform_c.text_to_int(utterance))
        labels.append(label)
        input_lengths.append(spec.shape[0])
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    
    #print(spectrograms.size())
#    
#    print(labels.size())

    return spectrograms, labels, input_lengths, label_lengths

def load(split, batch_size, workers=0, augmentation=False):
    """
    Args:
        split (string): Which of the subset of data to take. One of 'train', 'dev' or 'test'.
        batch_size (integer): Batch size.
        workers (integer): How many subprocesses to use for data loading.
        augmentation (bool): Apply SpecAugment to training data or not.

    Returns:
        loader (DataLoader): A DataLoader can generate batches of (FBANK features, FBANK lengths, label sequence).
    """
    assert split in ['train', 'dev', 'test']

    dataset = ASR(split, augmentation)
    # print(dataset)
    print ("%s set size:"%split.upper(), len(dataset))

    # kwargs = {'num_workers': 6, 'pin_memory': True} if use_cuda else {}

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=lambda x: data_processing_c(x, split),
                        num_workers=workers,
                        pin_memory=True)
    return loader

def loss_F(parameters):
    return sum(torch.linalg.norm(w) ** 2 for w in parameters)


def train(model, device, train_loader_e, train_loader_c, criterion, optimizer, grad_list, loss_list, multi_grad_fn,
                         kwargs, epoch, loss_dict,train_loader2,pre_criterion,pre_optimizer,gam):
    
    data_len = len(train_loader2)

    info_loss = 0

    model.train()
    
    train_loss = 0

    data_len_e = len(train_loader_e.dataset)
    data_len_c = len(train_loader_c.dataset)

    pre_optimizer.zero_grad()
    optimizer.zero_grad()

    # for batch_idx, (_data, inputs) in enumerate(zip(train_loader, train_loader2)):
    for batch_idx, (_data_e,_data_c,data_p) in enumerate(zip(train_loader_e,train_loader_c,train_loader2)):
            
            ########################       English #####################################################

            grad_list = []
            loss_list = [0 for _ in range(2)]

            context, future, negative_samples, lengths = data_p
            spectrograms_e, labels_e, input_lengths_e, label_lengths_e = _data_e
            spectrograms_c, labels_c, input_lengths_c, label_lengths_c = _data_c

            context = context.to(device)
            future = future.to(device)
            negative_samples = negative_samples.to(device)
            
            context = context.repeat(1, 80, 1)
            
            context = context.transpose(1,2)
            
#             print(context.size())
                
            input_lengths=torch.LongTensor(lengths).to(device)


            predictions = model(context, input_lengths)

            predictions = predictions[3]
            

            predictions = predictions[:, -1:, :]

            sizes = predictions.size()

            predictions = predictions.view(sizes[0], sizes[1]*sizes[2])

            target = future.view(sizes[0], sizes[1]*sizes[2])
            
            neg_target = negative_samples.view(sizes[0], sizes[1]*sizes[2])

            gam = round(gam, 3)
        
            # lamda = .001

            # reg =  loss_F(model.parameters())  #torch.norm(predictions)**2
    
            loss_cpc = gam*pre_criterion(predictions, target, neg_target) #+ lamda*reg  # gxy


            # model.classifier_zh.requires_grad = False

            # model.classifier_en.requires_grad = False

            # Backward and optimize
            
            # loss_cpc.backward()

            # if batch_idx % 8 == 0 or batch_idx == data_len:
            #     pre_optimizer.step()
            #     pre_optimizer.zero_grad()
            
            
            # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10, norm_type=2.0)
        
            info_loss += loss_cpc.item() / len(train_loader2)
        
            if batch_idx % 100 == 0 or batch_idx == data_len:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tCPC_Loss: {:.6f}'.format(
                    epoch, batch_idx * len(context), data_len,
                    50. * batch_idx / len(train_loader2), loss_cpc.item()))
                

            # model.classifier_zh.requires_grad = True

            # model.classifier_en.requires_grad = True


            # Get the actual batch sizes
            batch_size_e = spectrograms_e.size(0)
            batch_size_c = spectrograms_c.size(0)

            # Calculate the mid-points for each batch
            mid_e = batch_size_e // 2
            mid_c = batch_size_c // 2

            for i in range(2):
                if i == 0:
                    spectrograms_em = spectrograms_e[:mid_e]
                    labels_em = labels_e[:mid_e]
                    input_lengths_em = input_lengths_e[:mid_e]
                    label_lengths_em = label_lengths_e[:mid_e]

                    spectrograms_cm = spectrograms_e[mid_e:]
                    labels_cm = labels_e[mid_e:]
                    input_lengths_cm = input_lengths_e[mid_e:]
                    label_lengths_cm = label_lengths_e[mid_e:]

                    # print('english')


                elif i == 1:

                    spectrograms_em = spectrograms_c[:mid_c]
                    labels_em = labels_c[:mid_c]
                    input_lengths_em = input_lengths_c[:mid_c]
                    label_lengths_em = label_lengths_c[:mid_c]

                    spectrograms_cm = spectrograms_c[mid_c:]
                    labels_cm = labels_c[mid_c:]
                    input_lengths_cm = input_lengths_c[mid_c:]
                    label_lengths_cm = label_lengths_c[mid_c:]

                    # print('chinese')
            
                spectrograms_em = torch.squeeze(spectrograms_em)
                    
                # print(label_lengths_c.size())
                    
                spectrograms_em = spectrograms_em.transpose(1,2)
                    
                    # print(spectrograms.size())
                    
                labels_em= torch.LongTensor(labels_em.long())
                    
                input_lengths_em=torch.LongTensor(input_lengths_em)
                label_lengths_em=torch.LongTensor(label_lengths_em)
        #             print(label_lengths.type())
                input_lengths_em = input_lengths_em.to(device)
                label_lengths_em = label_lengths_em.to(device)
                spectrograms_em, labels_em = spectrograms_em.to(device), labels_em.to(device)
#             print(spectrograms.size())

  ########################################  Chinese ########################################################

            
            # print(spectrograms.shape)
            
                spectrograms_cm = torch.squeeze(spectrograms_cm)
                    
                    # print(spectrograms.size())
                    
                spectrograms_cm = spectrograms_cm.transpose(1,2)
                    
                    # print(spectrograms.size())
                    
                labels_cm= torch.LongTensor(labels_cm.long())
                    
                input_lengths_cm=torch.LongTensor(input_lengths_cm)
                label_lengths_cm=torch.LongTensor(label_lengths_cm)
        #             print(label_lengths.type())
                input_lengths_cm = input_lengths_cm.to(device)
                label_lengths_cm = label_lengths_cm.to(device)
                spectrograms_cm, labels_cm = spectrograms_cm.to(device), labels_cm.to(device)

                output_em, output_cm = model(spectrograms_em,input_lengths_em), model(spectrograms_cm,input_lengths_cm)  # (batch_size, sequence_length, dim)

                input_lengths_e1 = output_em[2]
                input_lengths_c1 = output_cm[2]
                
                 # (time, batch, n_class)

                if i==0:

                    output_e = output_em[0].transpose(0, 1) 
                    output_c = output_cm[0].transpose(0, 1) # (time, batch, n_class)

                elif i==1:

                    output_e = output_em[1].transpose(0, 1) 
                    output_c = output_cm[1].transpose(0, 1) # (time, batch, n_class)
            
                # loss_e = criterion(output_e, labels_em, input_lengths_e1, label_lengths_em)
                # loss_c = criterion(output_c, labels_cm, input_lengths_c1, label_lengths_cm)

       
                grad_list_, loss_list_ = get_grads(model, optimizer, num_param, num_param_layer, loss_dict,
                                                  output_e, labels_em, input_lengths_e1, label_lengths_em,
                                                 output_c, labels_cm, input_lengths_c1, label_lengths_cm) #####################  independent gradients \nabla f_{m,zt,s}(x_t)
                grad_list.append(grad_list_)
                # average the loss over two samples
                # loss_list = [loss_list[k] + 0.5 * loss_list_[k] for k in range(num_tasks)]
            # loss_cpc.backward()

            # if batch_idx % 8 == 0 or batch_idx == data_len_e:
            #     pre_optimizer.step()
            #     pre_optimizer.zero_grad()

            multi_grad, lambd = multi_grad_fn[moo_method](grad_list, **kwargs)
            # # update model grad with the multi-grad
            # set_grads(model, multi_grad, num_param_layer, device)

            # if batch_idx % 8 == 0 or batch_idx == data_len_e:
            #     optimizer.step()
            #     optimizer.zero_grad()




            spectrograms_e = torch.squeeze(spectrograms_e)
            
            # print(spectrograms.size())
            
            spectrograms_e = spectrograms_e.transpose(1,2)
            
            # print(spectrograms.size())
            
            labels_e= torch.LongTensor(labels_e.long())
            
            input_lengths_e=torch.LongTensor(input_lengths_e)
            label_lengths_e=torch.LongTensor(label_lengths_e)
# #             print(label_lengths.type())
            input_lengths_e = input_lengths_e.to(device)
            label_lengths_e = label_lengths_e.to(device)
            spectrograms_e, labels_e = spectrograms_e.to(device), labels_e.to(device)

            # print(spectrograms.shape)
            
            spectrograms_c = torch.squeeze(spectrograms_c)
            
            # print(spectrograms.size())
            
            spectrograms_c = spectrograms_c.transpose(1,2)
            
#             # print(spectrograms.size())
            
            labels_c= torch.LongTensor(labels_c.long())
            
            input_lengths_c=torch.LongTensor(input_lengths_c)
            label_lengths_c=torch.LongTensor(label_lengths_c)
# #             print(label_lengths.type())
            input_lengths_c = input_lengths_c.to(device)
            label_lengths_c = label_lengths_c.to(device)
            spectrograms_c, labels_c = spectrograms_c.to(device), labels_c.to(device)

            output_e, output_c = model(spectrograms_e,input_lengths_e), model(spectrograms_c,input_lengths_c)  # (batch_size, sequence_length, dim)

            input_lengths_e1 = output_e[2]
            input_lengths_c1 = output_c[2]
            
#             output = F.log_softmax(output, dim=2)
            output_e = output_e[0].transpose(0, 1) 
            output_c = output_c[1].transpose(0, 1) # (time, batch, n_class)

#             # print(output_c)
 
#            output = torch.tensor(output,  dtype=torch.float32).contiguous()
#            labels = torch.tensor(labels, dtype=torch.int)
#            output_lengths = torch.tensor(output_lengths, dtype=torch.int)
#            label_lengths = torch.tensor(label_lengths, dtype=torch.int)
            
            # optimizer.zero_grad()
            
            loss_e = criterion(output_e, labels_e, input_lengths_e1, label_lengths_e)
            loss_c = criterion(output_c, labels_c, input_lengths_c1, label_lengths_c)

            loss = loss_e*lambd[0]+ loss_c*lambd[1]+loss_cpc

            loss.backward(retain_graph=True)

            if batch_idx % 8 == 0 or batch_idx == data_len_e:
                pre_optimizer.step()
                pre_optimizer.zero_grad()
    
            train_loss += loss_c.item() / len(train_loader_c)


            loss_e_ctc = criterion(output_e, labels_e, input_lengths_e1, label_lengths_e)
            loss_c_ctc = criterion(output_c, labels_c, input_lengths_c1, label_lengths_c)

            #loss_ctc = loss_e_ctc*lambd[0]+ loss_c_ctc*lambd[1]
            
            loss_ctc = loss_e_ctc+ loss_c_ctc

            loss_ctc.backward()

            if batch_idx % 8 == 0 or batch_idx == data_len_e:
                optimizer.step()
                optimizer.zero_grad()

            if batch_idx % 100 == 0 or batch_idx == data_len_e:
      
                        print('Train Epoch English: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx * len(spectrograms_e), data_len_e,
                            100. * batch_idx / len(train_loader_e), loss_e.item()))  


            if batch_idx % 100 == 0 or batch_idx == data_len_c:

                            print('Train Epoch Chinese: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                epoch, batch_idx * len(spectrograms_c), data_len_c,
                                100. * batch_idx / len(train_loader_c), loss_c.item()))
    
                    
              

                
    return train_loss

def test(model, device, test_loader, criterion, epoch,language=1, batch_size=20):
    print('\nevaluating...')
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []

    if language==1:
        n_classes=1000
    else:
        n_classes = 5000

    if epoch%1==0:
        with torch.no_grad():
                for i, _data in enumerate(test_loader):
                    spectrograms, labels, input_lengths, label_lengths = _data
                    
                    spectrograms=torch.squeeze(spectrograms)
                    
                    spectrograms = spectrograms.transpose(1,2)
            
                    labels=labels.long()

                    input_lengths=torch.LongTensor(input_lengths)
                    label_lengths=torch.LongTensor(label_lengths)
                    input_lengths = input_lengths
                    label_lengths = label_lengths

                    spectrograms, labels = spectrograms.to(device), labels.to(device)

                    output_e, output_c, _ = model(spectrograms,input_lengths), model(spectrograms,input_lengths), model(spectrograms,input_lengths)  # (batch, time, n_class)

                    if language==1:
                        output=output_e[0]
                        output_lengths = output_e[2]

                    elif language==2:
                        output = output_c[1]

                        output_lengths = output_c[2]

#                     output_lengths = torch.full((output.size(0),), output.size(1), dtype=torch.int32)
#                     output = F.log_softmax(output, dim=2)
                    output = output.transpose(0, 1) # (time, batch, n_class)
                    
#                    output = torch.tensor(output,  dtype=torch.float32).contiguous()
#                    labels = torch.tensor(labels, dtype=torch.int)
#                    output_lengths = torch.tensor(output_lengths, dtype=torch.int)
#                    label_lengths = torch.tensor(label_lengths, dtype=torch.int)
                    
                    
                    loss = criterion(output, labels, output_lengths, label_lengths)
                    test_loss += loss.item() / len(test_loader)
                    
                    itera = spectrograms.size()

                    decoder = CTCBeamDecoder(
                        [''] * (n_classes - 1) + [' '],
                        model_path=None,
                        alpha=0,
                        beta=0,
                        cutoff_top_n=40,
                        cutoff_prob=1.0,
                        beam_width=100,
                        num_processes=4,
                        blank_id=0,
                        log_probs_input=True
                    )
                    print(output.shape)
                    beam_results, beam_scores, timesteps, out_lens = decoder.decode(output,output_lengths)
                    b=[]
                    for i in range(itera[0]):
                         b.append(beam_results[i][0][:out_lens[i][0]])
                    decoded_preds, decoded_targets = numtoword(b,out_lens,labels, label_lengths, language)
 
                    test_wer=calculate_wer(decoded_targets,decoded_preds)                

        avg_wer = sum(test_wer)/len(test_wer)

        print('Test set: Average loss: {:.4f}, Average WER: {:.4f}\n'.format(test_loss, avg_wer))
        
        # file_path = "/home/exx/Desktop/saif/conformer/wer.txt"
        # with open(file_path, "a") as file:
        #     file.write(f"Epoch {epoch}: {avg_wer}\n")

        return test_loss, avg_wer 
    #     return beam_results, out_lens, output ### set a counter in each iterration given the current update difference in abtch size can cause the update different 
    else:
        with torch.no_grad():
            for i, _data in enumerate(test_loader):
                spectrograms, labels, input_lengths, label_lengths = _data
                
                spectrograms=torch.squeeze(spectrograms)
                
                spectrograms = spectrograms.transpose(1,2)
            
                labels=labels.long()

                input_lengths=torch.LongTensor(input_lengths)
                label_lengths=torch.LongTensor(label_lengths)
                
                input_lengths = input_lengths.to(device)
                label_lengths = label_lengths.to(device)
                
                #print(spectrograms.size())
                

                spectrograms, labels = spectrograms.to(device), labels.to(device)

                output_e, output_c, _ = model(spectrograms,input_lengths), model(spectrograms,input_lengths), model(spectrograms,input_lengths)  # (batch, time, n_class)


                if language==1:
                        output=output_e[0]                        
                        output_lengths = output_e[2]

                elif language==2:
                        output = output_c[1]
                        output_lengths = output_c[2]


                output = output.transpose(0, 1) # (time, batch, n_class)
                
                
                loss = criterion(output, labels, output_lengths, label_lengths)
                test_loss += loss.item() / len(test_loader)
        print('Test set: Average loss: {:.4f}\n'.format(test_loss))
        return test_loss, 0 
      



def save_metrics(epoch, test_loss, wer):
    with open('modo_bilevel.txt', 'a') as file:
        file.write(f'Epoch {epoch}: Test Loss = {test_loss:.4f}, WER = {wer:.4f}\n')


def get_audio_files_flac(data_dir):
    return [os.path.join(root, file) for root, dirs, files in os.walk(data_dir) for file in files if file.lower().endswith('.flac')]

def get_audio_files_wav(data_dir):
    return [os.path.join(root, file) for root, dirs, files in os.walk(data_dir) for file in files if file.lower().endswith('.wav')]


if __name__== "__main__":

    learning_rate = 5e-4
    batch_size = 16
    epochs = 100
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
    moo_method='MoDo'

    modo_gamma=.1

    modo_rho=0.0

    hparams = {

        "n_class": 5000,
        "n_feats": 80,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs
    }
    train_url="train-clean-100"
    test_url="test-clean"

    # experiment.log_parameters(hparams)

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")

      
    kwargsd = {'num_workers': 6, 'pin_memory': True} if use_cuda else {}
    train_loader_c = load('train', 32)
    test_loader_c = load('test', 50)

    if not os.path.isdir("./data"):
        os.makedirs("./data")

    splits = ["train-clean-100", "train-clean-360", "train-other-500"]

    train_dataset1 = torchaudio.datasets.LIBRISPEECH("./data", url=splits[0], download=True)
    # train_dataset22 = torchaudio.datasets.LIBRISPEECH("./data", url=splits[1], download=True)
    # train_dataset3 = torchaudio.datasets.LIBRISPEECH("./data", url=splits[2], download=True)
    # # Combine the dataset splits into a single dataset
    # combined_dataset = data.ConcatDataset([train_dataset1, train_dataset22, train_dataset3])

    # splits = ["train-clean-100", "train-clean-360", "train-other-500"]
        
    splits = ["train-clean-100", "train-clean-360"]

# Load datasets dynamically using a loop
    datasets = [torchaudio.datasets.LIBRISPEECH("./data", url=split, download=True) for split in splits]

    # Combine the datasets into a single dataset
    combined_dataset = data.ConcatDataset(datasets)

    # train_dataset = torchaudio.datasets.LIBRISPEECH("./data", url=train_url, download=True)

    test_dataset = torchaudio.datasets.LIBRISPEECH("./data", url=test_url, download=True)

    # combined_dataset = data.ConcatDataset([train_loader,train_dataset])

    # combined_testset = data.ConcatDataset([test_loader,test_dataset])

    train_loader_e = data.DataLoader(dataset = train_dataset1,
                                batch_size=hparams['batch_size'],
                                shuffle=True,
                                collate_fn=lambda x: data_processing(x, 'train'),
                                **kwargsd)
    test_loader_e = data.DataLoader(dataset=test_dataset,
                                batch_size= 50,#hparams['batch_size'],
                                shuffle=False,
                                collate_fn=lambda x: data_processing(x, 'valid'),
                                **kwargsd)

    model = conformer(num_classes=hparams['n_class'], 
                  input_dim=hparams['n_feats'], 
                  num_encoder_layers=8)
    
    model = nn.DataParallel(model)

#     print(model)

    
    model.load_state_dict(torch.load('/media/chenlab2/hdd5/saif/asr/modo_just960.pth'))
    
    model.to(device)
    
    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    num_param, num_param_layer = get_layer_params(model)
    
    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
    criterion = nn.CTCLoss(blank=0).to(device)

    # criterion_e = nn.CTCLoss(blank=0).to(device)
    # criterion_c = nn.CTCLoss(blank=0).to(device)

    # loss_fn = InfoNCE()

    loss_dict = {'eng':criterion, 'chn':criterion}
    # number of tasks
    num_tasks = 2  #len(loss_dict)

    multi_grad_fn = {'MoDo':grad_modo}
    modo_kwargs = {'lambd':torch.ones(num_tasks)/num_tasks, 'gamma':modo_gamma, 'rho':modo_rho}
    kwargs = {'MoDo':modo_kwargs}
    
    #criterion = torchaudio.transforms.RNNTLoss().to(device)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'], 
                                            steps_per_epoch=int(len(train_loader_c)),
                                            epochs=hparams['epochs'],
                                            anneal_strategy='linear')
#    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=5e-4, verbose=True)

####################################Pre training######################################
    
    
    data_dir_c = "/media/chenlab2/hdd5/saif/asr/asr/conformer/conformer/data_aishell/wav"
    data_dir_e = "./data/LibriSpeech"
    
    # audio_files = []

    # for root, dirs, files in os.walk(data_dir_c):
    #     for file in files:
    #        if file.lower().endswith('.wav'):
    #             audio_files.append(os.path.join(root, file))

    # for root, dirs, files in os.walk(data_dir_e):
    #     for file in files:

    #         if file.lower().endswith('.flac'):  # Adjust the file extension filter as needed
    #             audio_files.append(os.path.join(root, file))

    audio_files = get_audio_files_wav(data_dir_c) + get_audio_files_flac(data_dir_e)


    waveform_length = 16000  # Length of the waveform (can be adjusted as needed)
    context_length = 256  # Length of the context wave
    future_length = 100  # Length of the future samples
    negative_waveform_length = 100

    train_dataset2 = LibriSpeechDataset(audio_files, waveform_length, context_length, future_length, negative_waveform_length)
      # Adjust the batch size as needed
    train_loader2 = DataLoader(train_dataset2, batch_size=32,shuffle=True) # Iterate over the data loader
    
   
    pre_optimizer = optim.AdamW(model.parameters(), lr=5e-3)
    pre_scheduler = optim.lr_scheduler.OneCycleLR(pre_optimizer, max_lr=.001, 
                                            steps_per_epoch=int(len(train_loader2)),
                                            epochs=hparams['epochs'],
                                            anneal_strategy='linear')

    pre_criterion = InfoNCE()


    gamma_max = 1
    gamma_init = 0
    gamma_argmax_step = 500
    if gamma_init > gamma_max:
        gamma_max = gamma_init
        print('Initial gamma is larger than max gamma, proceeding with gamma_max=gamma_init.')
    gam = gamma_init
    # step_gam = (gamma_max-gamma_init)/gamma_argmax_step
    step_gam = (gamma_max-0)/gamma_argmax_step

    print(len(train_loader2.dataset))


    if not os.path.isfile('modo_bilevel_960.txt'):
        with open('modo_bilevel960.txt', 'w') as file:
            file.write("Test Loss for English and Test Loss for Chinese\n")


    train_loss=[]
    test_loss=[]
    test_loss_c=[]
    cer=[]
    wer=[]
    wer_c=[]
    test_loss1=6

    
    start_time = time.time()

    for epoch in range(1, epochs + 1):

        grad_list = []
        loss_list = [0 for _ in range(num_tasks)]
        
        
        gam = min(gamma_max,gam)

        # tra_loss = train(model, device, train_loader_e, train_loader_c, criterion, optimizer, grad_list, loss_list, multi_grad_fn,
        #                  kwargs, epoch, loss_dict,train_loader2,pre_criterion,pre_optimizer,gam)
        
        gam+= step_gam
        
        tes_loss, w =  test(model, device, test_loader_e, criterion, epoch, 1)
        tes_loss_c, w_c =  test(model, device, test_loader_c, criterion, epoch, 2)
        # scheduler.step()
        # pre_scheduler.step()
        # train_loss.append(tra_loss)
        # test_loss.append(tes_loss)
        # test_loss_c.append(tes_loss_c)
        # wer.append(w)
        # wer_c.append(w_c)

        # if tes_loss<test_loss1 or tes_loss_c<test_loss1:
        #         if tes_loss<tes_loss_c:
        #             test_loss1 = tes_loss
        #         elif tes_loss>tes_loss_c:
        #             test_loss1 = tes_loss_c
        #         torch.save(model.state_dict(), './conformer/modo_just960.pth')
        #         print('saved!!!!!!')

        # print(f'Best test loss:{test_loss1}')

        # with open('modo_bilevel_960.txt', 'a') as file:
        #     file.write(f'Test for English Loss over {epoch} epochs = {tes_loss:.4f}\n')
        #     file.write(f'Test for Chinese Loss over {epoch} epochs = {tes_loss_c:.4f}\n')
        #     file.write(f'Gamma {epoch} epochs = {gam:.4f}\n')

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print the time taken for one epoch
    print(f"Time taken for one epoch: {elapsed_time} seconds")
