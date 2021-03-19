#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import os
import argparse
import scipy.io as sio
import sys
from ge2e import GE2ELoss
import deepdish as dd

def load_data_short(full_data_name, max_speakers):
    f=h5py.File(full_data_name, 'r')
    data_ref=f['data']
    #label_ref=f['labels']
    nSpeaker=data_ref.shape[1]
    if nSpeaker>max_speakers and max_speakers!=0:
        nSpeaker=max_speakers
    #labels=[]
    data=[]
    check=np.array(list(range(1,100)))
    check_ind=0
    for spk in range(nSpeaker):       
        data2=[]
        nChannel=f[data_ref[0,spk]].shape[1]
        if nChannel>=8:
            for chan in range(nChannel):
                data3=torch.from_numpy(f[f[data_ref[0,spk]][0,chan]][:])
                data2.append(data3.float())
            data.append(data2)
        #label=''.join([chr(f[label_ref[0,spk]][i]) for i in range(7)])
        #labels.append(''.join(label))
        if check_ind<len(check):
            if (spk/nSpeaker)*100>=check[check_ind]:
                print('Speakers processed spk/nSpeaker = '+str(spk)+'/'+str(nSpeaker))
                check_ind+=1
    f.close() 
    
    #return data, labels
    return data

def load_data3(full_data_name, max_speakers, min_T, max_T, outputfile):
    #data, labels=load_data_short(full_data_name)
    data = load_data_short(full_data_name, max_speakers)
    print('Data loading complete, forming data!')
    T=[]
    nChannel=[]
    nceps=len(data[0][0][0])
    
    for spk in range(len(data)):
        for chan in range(len(data[spk])):
            T.append(data[spk][chan].shape[0])
        nChannel.append(len(data[spk]))
    if max_T>0 and min_T>0 and max_T>=min_T:
        T.append(max_T)
        T=np.min(T)
        #T.append(min_T)
        T=np.max([T, min_T])
        T=int(T)
    else:
        T=int(np.min(T))
    target_range=int(np.min(nChannel))
    
    fs=16000
    # MFCC was calculated with stFs=spectrogram(filter([1 -0.97], 1, audio), hamming(400), 240);
    # so 1 sample = 8 ms according to https://www.kfs.oeaw.ac.at/manual/3.8/html/userguide/464.htm
    nSpeaker=len(data)
    #minibatch_size=64
    # data for nn in shape NxCxTxembedding, where C is number of classes=speakers
    max_batch=target_range
    minibatch_size=nSpeaker
    #max_batch=int(np.floor(target_range*nSpeaker/minibatch_size))
    #minibatch_size=int(np.floor(target_range*nSpeaker/max_batch))
    if minibatch_size==0:
        max_batch=1
        minibatch_size=nSpeaker
    data_out=torch.zeros([max_batch,minibatch_size,T,nceps], dtype=torch.float)
    labels_out=torch.zeros([max_batch,minibatch_size], dtype=torch.long)
    print('data_out.shape = '+str(data_out.shape))
    print('labels_out.shape = '+str(labels_out.shape))
    ind1=0
    ind2=0
    check=np.array(list(range(1,100)))
    check_ind=0
    spk_ind=0
    for spk in range(nSpeaker):
        if len(data[spk])>=target_range:
            for c in range(target_range):
                if ind1>=max_batch:
                    continue
                if data[spk][c].shape[0]>=T:
                    #data_out[ind1,ind2,:,:]=torch.tensor(data[spk][c][:T,:], dtype=torch.double)
                    for t in range(T):
                        for n in range(nceps):
                            data_out[ind1,ind2,t,n]= data[spk][c][t,n].clone().detach().float()
                    labels_out[ind1,ind2]=spk_ind
                    ind2+=1
                    if ind2>=minibatch_size:
                        ind1+=1
                        ind2=0
            spk_ind+=1
        if check_ind<len(check):
            if (spk/nSpeaker)*100>=check[check_ind]:
                print('Speakers processed spk/nSpeaker = '+str(spk)+'/'+str(nSpeaker))
                check_ind+=1
    
    speakers=torch.unique(labels_out)
    max_speaker=speakers.max()
    if len(speakers)==len(torch.tensor(list(range(max_speaker+1)))):
        if ((speakers==torch.tensor(list(range(max_speaker+1))))+0.0).mean().item()==1:
            print('Labels >=0 and len(labels)<num_classes before shuffling')
        else:
            print('Incorrect labels before shuffling')
            labels_out=fix_labels(labels_out)
    else:
        print('Incorrect labels before shuffling')
        labels_out=fix_labels(labels_out)
            
    return shuffle_data2(data_out, labels_out, outputfile)

def shuffle_data(data, labels):
    data_out=torch.zeros(data.shape, dtype=data.dtype)
    labels_out=torch.zeros(labels.shape, dtype=labels.dtype)
    #ind=torch.tensor(list(range(data_out.shape[1])),dtype=torch.long)
    for ind in range(data.shape[0]):
        new_ind=torch.randperm(data_out.shape[1])
        for spk in range(data.shape[1]):
            for t in range(data.shape[2]):
                for n in range(data.shape[3]):
                    data_out[ind,spk,t,n]= data[ind,new_ind[spk],t,n].clone().detach()
            labels_out[ind,spk]=labels[ind,new_ind[spk]].clone().detach()
    return data_out, labels_out

def fix_labels(labels):
    speakers=np.unique(labels)
    max_speaker=int(np.max(speakers))

    for spk in range(len(speakers)):
        while len(np.where(spk==labels)[0])==0:
            ix,jx=np.where(labels>spk)
            ind=len(ix)
            for i in range(ind):
                labels[ix[i],jx[i]]-=1
                fixed_ind+=1
    print('Fixed '+str(fixed_ind)+' amount of labels')
    return labels

def shuffle_data2(data, labels, outputfile):
    data_out=torch.zeros(data.shape, dtype=data.dtype)
    labels_out=torch.zeros(labels.shape, dtype=labels.dtype)
    new_ind=torch.randperm(data_out.shape[0])
    check_ind=1
    for ind in range(data.shape[0]):
        for spk in range(data.shape[1]):
            for t in range(data.shape[2]):
                for n in range(data.shape[3]):
                    data_out[ind,spk,t,n]= data[new_ind[ind],spk,t,n].clone().detach()
            labels_out[ind,spk]=labels[new_ind[ind],spk].clone().detach()
        if (ind/data.shape[0])>=check_ind/10:
            print('Batch processed batch/nBatch = '+str(ind)+'/'+str(data.shape[0]))
            check_ind+=1
    
    #f = h5py.File('myfile.hdf5','w')
    #dset = f.create_dataset("init", {"data":data_out.numpy(), "labels":labels_out.numpy()})
    dd.io.save(outputfile, {"data":data_out.numpy(), "labels":labels_out.numpy()}, compression=('blosc', 9))
    #f.close()
    print("Data saved, lets check if file still exists")
    bar = dd.io.load(outputfile)
    labels=bar["labels"]
    data=bar["data"]
    print('labels.shape='+str(labels.shape))
    print('data.shape='+str(data.shape))
    #return data_out, labels_out

def isnan(x):
    return x!=x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='X-vector calculator')
    parser.add_argument('dataFilename', metavar='dataFilename', type=str, help='name of data as matfile')
    parser.add_argument('outputfile', metavar='outputfile', type=str, help='name of file for formed data')
    parser.add_argument('max_speaker', metavar='max_speaker', type=int, help='Desired numper of speakers')
    args = parser.parse_args()
    dataFilename=args.dataFilename
    max_speaker=args.max_speaker
    outputfile=args.outputfile
    #print("Device is",device)
    load_data3(dataFilename, max_speaker, 400, 400, outputfile)

    print("Data loaded and saved in new form!")
