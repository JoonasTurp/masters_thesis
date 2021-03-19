#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import os
import argparse
import scipy.io as sio
#from ge2e import GE2ELoss
import deepdish as dd


class TDNN(nn.Module):
    
    def __init__(
                    self, 
                    input_dim=23, 
                    output_dim=512,
                    context_size=5,
                    stride=1,
                    dilation=1,
                    batch_norm=True,
                    dropout_p=0.0
                ):
        '''
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf

        Affine transformation not applied globally to all frames but smaller windows with local context

        batch_norm: True to include batch normalisation after the non linearity
        
        Context size and dilation determine the frames selected
        (although context size is not really defined in the traditional sense)
        For example:
            context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
            context size 3 and dilation 2 is equivalent to [-2, 0, 2]
            context size 1 and dilation 1 is equivalent to [0]
        '''
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
      
        self.kernel = nn.Linear(input_dim*context_size, output_dim)
        self.nonlinearity = nn.ReLU()
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
        if self.dropout_p:
            self.drop = nn.Dropout(p=self.dropout_p)
        
    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)onn
        outpu: size (batch, new_seq_len, output_features)
        '''

        x = x.unsqueeze(1)

        # Unfold input into smaller temporal contexts
        x = F.unfold(
                        x, 
                        (self.context_size, self.input_dim), 
                        stride=(1,self.input_dim), 
                        dilation=(self.dilation,1)
                    )

        # N, output_dim*context_size, new_t = x.shape
        x = x.transpose(1,2)
        x = self.kernel(x)
        x = self.nonlinearity(x)
        
        if self.dropout_p:
            x = self.drop(x)

        if self.batch_norm:
            x = x.transpose(1,2)
            x = self.bn(x)
            x = x.transpose(1,2)

        return x

def stats_layer(x):
    # input dim = [x1,T,1500]
    # output_dim = [x1,1,T*1500]
    x=torch.cat((x.mean(dim=1),x.std(dim=1)),1)
    return x.unsqueeze(1)

class xvector_network(nn.Module):
    def __init__(self, nceps):
        super(xvector_network, self).__init__()
        self.layer1 = TDNN(input_dim=nceps, output_dim=512, context_size=5, dilation=1)
        self.layer2 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=2)
        self.layer3 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=3)
        self.layer4 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1)
        self.layer5 = TDNN(input_dim=512, output_dim=1500, context_size=1, dilation=1)
        self.segment6 = TDNN(input_dim=3000, output_dim=512, context_size=1, dilation=1)
        #self.segment7=TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1)
            
    def forward(self, input):
        x=self.layer1(input)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.layer5(x)
        #print(x.shape)
        x=stats_layer(x)
        #print(x.shape)
        x=self.segment6(x)
        #x=self.segment7(x)
        return x

class xvector_trainer(nn.Module):
    def __init__(self):
        super(xvector_trainer, self).__init__()
        self.segment7=TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1)
            
    def forward(self, input):
        x=self.segment7(input)
        return x

def load_data(full_data_name,min_T,max_T):
    f=h5py.File(full_data_name, 'r')
    ref=f['data'][0]
    nSpeaker=ref.shape[0]
    #nSpeaker=2
    data=[]
    a=[]
    
    for spk in range(nSpeaker):       
        n=f[ref[spk]].shape[1]
        data2=[]
        for chan in range(n):
            ref2=f[ref[spk]][0,chan]
            data3=torch.tensor(f[ref2])
            if len(data3)>min_T:
                if np.floor(len(data3)/max_T)>=2:
                    split=int(np.floor(len(data3)/max_T))
                    for s in range(split):
                        l=np.floor(s*max_T)
                        l2=np.floor((s+1)*max_T)
                        if s==split-1:
                            data2.append(data3[int(l):,:])
                        else:
                            data2.append(data3[int(l):int(l2),:])
                else:
                    data2.append(data3)
        data.append(data2)
    return data

def form_data(data):
    nSpeaker=len(data)
    data_out=[]
    nceps=len(data[0][0][0])
    for ix in range(nSpeaker):
        nChannel=int(len(data[ix]))
        m=np.array([data[ix][jx].shape[0] for jx in range(nChannel)])
        T=int(np.min(m))
        data_out2=torch.zeros(nChannel,T,nceps)
        for jx in range(nChannel):
            data_out2[jx]=data[ix][jx][:T,:]
        data_out.append(data_out2)
    return data_out          

def shuffle_data(data):
    nSpeaker=len(data)
    data_out=[]
    nceps=len(data[0][0][0])
    nChannel=torch.ones((nSpeaker,))
    T_lens=torch.ones((nSpeaker,))
    for ix in range(nSpeaker):
        nChannel[ix]=int(len(data[ix]))
        m=np.array([data[ix][jx].shape[0] for jx in range(int(nChannel[ix]))])
        T_lens[ix]=int(np.min(m))
        data_out2=torch.zeros(int(nChannel[ix]),int(T_lens[ix]),nceps)
        for jx in range(int(nChannel[ix])):
            data_out2[jx]=data[ix][jx][:int(T_lens[ix]),:]
        data_out.append(data_out2)
        del data_out2
    C=int(torch.min(nChannel))
    T=int(torch.min(T_lens))
    
    cChunk=torch.floor(nChannel/C)
    tChunk=torch.floor(T_lens/T)
    nbatch=int(torch.sum(tChunk*cChunk))
    labels=torch.tensor(list(range(nSpeaker)))
    data_out_final=torch.zeros((nbatch,C,T,nceps))
    labels_final=torch.zeros((nbatch,C),dtype=torch.long)

    batch_ind=0
    done=0
    for spk in range(nSpeaker):
        d=data_out[spk]
        #d2=torch.chunk(d, int(cChunk[spk]), dim=0)
        for ix in range(int(cChunk[spk])):
            c_ind=list(range(ix*C,(ix+1)*C))
            d2=d[c_ind]
            #d3=torch.chunk(d2[ix], int(tChunk[spk]), dim=1)
            for jx in range(int(tChunk[spk])):
                t_ind=list(range(jx*T,(jx+1)*T))
                data_out_final[batch_ind,:,:,:]=d2[:,t_ind,:]
                #data_out_final[batch_ind,:,:,:]=d3[jx][:C,:T,:]
                labels_final[batch_ind,:]=spk
                batch_ind+=1

    #shuffle indexes:
    shuffled_data=0*data_out_final
    shuffled_labels=0*labels_final
    ind=torch.tensor(list(range(nbatch)),dtype=torch.long)
    for c in range(C):
        new_ind=torch.randperm(nbatch)
        shuffled_data[ind,c]=data_out_final[new_ind,c]
        shuffled_labels[:,c]=labels_final[new_ind,c]
    
    return shuffled_data, shuffled_labels

def xvector_init(weigthFilename, model):
    pretrained_dict = torch.load(weigthFilename, map_location=torch.device('cpu'))
    model_dict = model.state_dict()
    if len(model_dict)==len(pretrained_dict):
        model.load_state_dict(pretrained_dict)
    else:
        # 1. filter out unnecessary keys
        pretrained_dict_partial = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict_partial) 
        # 3. load the new state dict
        model.load_state_dict(model_dict)    
    return model

def cosine_distance(v,u):
    return np.dot(v,u)/(np.linalg.norm(v)*np.linalg.norm(u))

def cosine_scoring(xvectors):
    scores=np.zeros((xvectors.shape[0],xvectors.shape[0]))
    for ix in range(xvectors.shape[0]):
        for jx in range(xvectors.shape[0]):
            scores[ix,jx]=cosine_distance(xvectors[ix],xvectors[jx])
    return scores

def prediction(xvectors,models):
    prediction=np.zeros((xvectors.shape[0],1))
    for ix in range(xvectors.shape[0]):
        scores=np.zeros((models.shape[0],1))
        for jx in range(models.shape[0]):
            scores[jx]=cosine_distance(xvectors[ix],models[jx])
        prediction[ix]=np.argmax(scores)
    return prediction 

def accuracy(prediction,labels):
    return np.sum(prediction==labels)/len(labels)*100

def x_vector(dataFilename, weigthFilename, outputFilename, max_speaker, device):
    print("Loading data")
    min_T, max_T=400, 400
    if os.path.isfile(dataFilename):
        bar = dd.io.load(dataFilename)
        data=torch.from_numpy(bar["data"]).float().contiguous()
        labels=torch.from_numpy(bar["labels"]).long().contiguous()
        del bar
    nceps=data[0].shape[2]
    print('Data loaded')
    model=xvector_network(nceps)
    

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model=nn.DataParallel(xvector_init(weigthFilename, model)).float().to(device)
    else:
        model=xvector_init(weigthFilename, model).float().to(device)
    
    model_output_size=512
    #xvectors=np.zeros((labels.shape[0]*labels.shape[1],model_output_size),dtype=np.double)
    #labels_final=np.zeros((labels.shape[0]*labels.shape[1],1),dtype=np.double)
    xvectors=[]
    labels_final=[]
    print("Initiation complete, calculating x-vectors from model")
    check=np.array(list(range(1,100)))
    check_ind=0
    file_ind=1
    if data.shape[1]>128:
        miniminibatch=128
    else:
        miniminibatch=data.shape[1]
    with torch.no_grad():
        ind=0
        for ix in range(data.shape[0]):
            for i in range(int(np.floor(data.shape[1]/miniminibatch))):
                output=model(data[ix,i*miniminibatch:(i+1)*miniminibatch].float().to(device)).squeeze()
                output=output.cpu().numpy()
                l=labels[ix,i*miniminibatch:(i+1)*miniminibatch]
                for jx in range(output.shape[0]):
                    #xvectors[ind,:]=output[jx,:]
                    #labels_final[ind,0]=labels[ix,jx]
                    #ind+=1
                    xvectors.append(output[jx,:])
                    labels_final.append(l[jx])
                del output,l
            if int(np.floor(data.shape[1]/miniminibatch))*miniminibatch<data.shape[1]:
                output=model(data[ix,int(np.floor(data.shape[1]/miniminibatch))*miniminibatch:-1].float().to(device)).squeeze()
                output=output.cpu().numpy()
                l=labels[ix,int(np.floor(data.shape[1]/miniminibatch))*miniminibatch:-1]
                for jx in range(output.shape[0]):
                    #xvectors[ind,:]=output[jx,:]
                    #labels_final[ind,0]=labels[ix,jx]
                    #ind+=1
                    xvectors.append(output[jx,:])
                    labels_final.append(l[jx])
                del output,l
            if check_ind<len(check):
                if (ix/len(data))*100>=check[check_ind]:
                    print('Speakers processed spk/nSpeaker = '+str(ix)+'/'+str(len(data)))
                    check_ind+=1
            if ix/len(data)>=file_ind/3 and len(data)>=100:
                labels_final=np.array(labels_final)
                xvectors=np.array(xvectors)
                #score,answers=cosine_scoring(xvectors, labels)
                if os.path.isfile(outputFilename+'_'+str(file_ind)+'.mat'):
                    os.system('rm '+outputFilename+'_'+str(file_ind)+'.mat')
                sio.savemat(outputFilename+'_'+str(file_ind)+'.mat',{'xvectors' : xvectors, 'labels' : labels_final})
                xvectors=[]
                labels_final=[]
                file_ind+=1
    
    labels_final=np.array(labels_final)
    xvectors=np.array(xvectors)
    #score,answers=cosine_scoring(xvectors, labels)
    if os.path.isfile(outputFilename+'_'+str(file_ind)+'.mat'):
        os.system('rm '+outputFilename+'_'+str(file_ind)+'.mat')
    sio.savemat(outputFilename+'_'+str(file_ind)+'.mat',{'xvectors' : xvectors, 'labels' : labels_final})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='X-vector calculator')
    parser.add_argument('dataFilename', metavar='dataFilename', type=str, help='name of data as matfile')
    parser.add_argument('weigthFilename', metavar='weigthFilename', type=str, help='Location of network model')
    parser.add_argument('outputFilename', metavar='validation_dataFilename', type=str, help='Desired location of validation data')
    parser.add_argument('max_speaker', metavar='max_speaker', type=int, help='Desired numper of speakers')
    args = parser.parse_args()
    dataFilename=args.dataFilename
    weigthFilename=args.weigthFilename
    outputFilename=args.outputFilename
    max_speaker=args.max_speaker

    print("Loading data from "+dataFilename+"\n"
          +"Loading weigths in "+weigthFilename+"\n"
          +"Storing scoredata in "+outputFilename+"\n")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print("Device is",device)
    x_vector(dataFilename, weigthFilename, outputFilename, max_speaker, device)

    print("X-vectors calculated!")



