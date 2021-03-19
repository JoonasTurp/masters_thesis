import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import os
import argparse


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
    
def load_data(full_data_name, max_speakers):
    
    datafile = h5py.File(full_data_name, 'r')
    myref=datafile["train_data"][0]
    nspkears=len(myref)
    if (nspkears>max_speakers) and (max_speakers!=0):
        nspkears=max_speakers
    n=np.zeros(nspkears,)
    for i in range(nspkears):
        n[i] = datafile[myref[i]].shape[0]
        
    nceps=int(datafile[myref[0]].shape[-1])
    
    target_range=int(np.min(n))
    fs=16000
    # MFCC was calculated with stFs=spectrogram(filter([1 -0.97], 1, audio), hamming(400), 240);
    # so 1 sample = 8 ms according to https://www.kfs.oeaw.ac.at/manual/3.8/html/userguide/464.htm
    
    minibatch_size=64
    max_chunk_size=int(np.floor(target_range/minibatch_size))
    
    data=torch.zeros(minibatch_size,nspkears,max_chunk_size,nceps)
    labels=torch.LongTensor(minibatch_size,nspkears)
    
    for i in range(minibatch_size):
        for j in range(nspkears):
            data[i,j,:,:]=torch.tensor(datafile[myref[j]][i*max_chunk_size:(i+1)*max_chunk_size,:])
            labels[i,j]=j
            
    return data, labels

def load_data4(full_data_name,min_T,max_T):
    f=h5py.File(full_data_name, 'r')
    ref=f['data'][0]
    nSpeaker=ref.shape[0]
    data=[]
    a=[]
    
    for spk in range(nSpeaker):       
        n=f[ref[spk]].shape[1]
        data2=[]
        for chan in range(n):
            ref2=f[ref[spk]][0,chan]
            data3=np.array(f[ref2])
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
    return form_data2(data)

def load_data2(full_data_name,min_T,maxT):
    f=h5py.File(full_data_name, 'r')
    ref=f['data'][0]
    nSpeaker=ref.shape[0]
    data=[]
    a=[]
    
    for spk in range(nSpeaker):       
        n=f[ref[spk]].shape[1]
        data2=[]
        for chan in range(n):
            ref2=f[ref[spk]][0,chan]
            data3=np.array(f[ref2])
            if data3.shape[0]>min_T:
                if data3.shape[0]>maxT:
                    data2.append(data3[:maxT])
                else:
                    data2.append(data3)
            else:
                if len(a)+len(data3)>min_T:
                    data2.append(np.concatenate((a,data3)))
                    a=[]
                    if len(a)+len(data3)>maxT:
                        data4=np.concatenate((a,data3))
                        data2.append(data4[:maxT])
                        del data4
                    else:
                        data2.append(np.concatenate((a,data3)))
                        a=[]
                elif len(a)>0:
                    a=np.concatenate((a,data3))
                else:
                    a=data3
        data.append(data2)
    return form_data2(data)

def form_data2(data):
    nSpeaker=len(data)
    data_out=[]
    nceps=len(data[0][0][0])
    for ix in range(nSpeaker):
        nChannel=int(len(data[ix]))
        m=np.array([data[ix][jx].shape[0] for jx in range(nChannel)])
        T=int(np.min(m))
        data_out2=torch.zeros(nChannel,T,nceps)
        for jx in range(nChannel):
            data_out2[jx]=torch.tensor(data[ix][jx][:T,:])
        data_out.append(data_out2)
    return data_out

def get_paths(path,name):
    date="oct30"
    if name=="dataFilename":
        filename=path+'train_data-oct30-10.mat'
    elif name=="weigthFilename":
        filename=path+"xvector_model"+date+"_v2.pth"
        if os.path.isfile(filename):
            i=3
            done=0
            while done==0:
                filename=path+"xvector_model"+date+"_v"+str(i)+".pth"
                if os.path.isfile(filename)==0:
                    done=1
                else:
                    i=i+1           
    return filename

# Main algorithm:
def x_vector_trainer(dataFilename, weigthFilename):
    print("Loading data")
    x=0
    try:
        data, labels = load_data(dataFilename,0)
        print("Data load complete")
        print("Data size is",data.shape)
        x=1
        batch_size, nspkr, _, nceps = data.shape
    except:
        try:
            data = load_data2(dataFilename)
            nspkr=len(data)
            labels=[]
            for spk in range(nspkr):
                labels.append(spk*torch.ones(1,data[spk].shape[0]))
            print("Data load complete with load_data2")
            x=2
        except:
            print("Data load failed")
    
    if x==1:
        # Training xvectors with constant batch size
        model=xvector_network(nceps)
        trainer=xvector_trainer()
        print("Model is:")
        print(model)
        criterion=nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.003)
        
        epochs = 10
        print("Initiation complete")
        
        print("Training x-vector model")
        for e in range(epochs):
            running_loss = 0.0
            for i in range(int(batch_size)):
                optimizer.zero_grad()
                output=trainer(model(data[i]))
                loss=criterion(torch.squeeze(output),labels[i])
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print('epoch: %d/%d calculated with loss: %.7f' %(e + 1,epochs, running_loss / batch_size))
            
        print("Finished Training, saving model!")
        torch.save(model.state_dict(), weigthFilename)
        print("X-vector model training complete!")
            
    else:
        print("No data to process")
        
    
def x_vector_trainer2(dataFilename, weigthFilename):
    print("Loading data")
    x=0
    try:
        min_T = 200
        max_T= 400
        data=load_data4(full_data_name,min_T,max_T)
        #data=form_data3(data)
        print("Data load complete with load_data4")
        print(data[0].shape)
        x=2
    except:
        print("Data load failed")
    
    if x==2:
        # Training xvectors with variable batch size
        nceps=data[0].shape[2]
        nspkr=len(data)
        model=xvector_network(nceps)
        trainer=xvector_trainer()
        criterion=nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.003)
        
        epochs = 5
        print("Initiation complete")
        
        print("Training x-vector model")
        for e in range(epochs):
            running_loss = 0.0
            for i in range(int(nspkr)):
                optimizer.zero_grad()
                output=trainer(model(data[i]))
                t1,_,t2=output.shape
                #output=trainer(model(data[i]))
                labels=i*torch.ones(t1,t2,dtype=torch.long)
                loss=criterion(output,labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                print("Spkeaker",i,"processed in epoch",e+1)
            print('epoch: %d/%d calculated with loss: %.7f' %(e + 1,epochs, running_loss / nspkr))
        
        print("Finished Training, saving model!")
        torch.save(model.state_dict(), weigthFilename)
        print("X-vector model training complete!")
            
    else:
        print("No data to process")       
        
if __name__ == '__main__':
    current_path = os.getcwd()
    current_path=current_path+"/"
    
    dataFilename=get_paths("/scratch/work/turpeim1/matlab/data/","dataFilename")
    weigthFilename=get_paths(current_path,"weigthFilename")
        
    print("Loading data from "+dataFilename+"\n"
          +"Storing weigths in "+weigthFilename+"\n")
    
    # Actual algorithm:
    try:
        print("Loading data")
        x=0
        try:
            min_T = 200
            max_T= 400
            data=load_data4(full_data_name,min_T,max_T)
            #data=form_data3(data)
            print("Data load complete with load_data4")
            print(data[0].shape)
            x=2
        except:
            print("Data load failed")
            
        try:
            if x==2:
                # Training xvectors with variable batch size
                nceps=data[0].shape[2]
                nspkr=len(data)
                model=xvector_network(nceps)
                trainer=xvector_trainer()
                criterion=nn.CrossEntropyLoss()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.003)
                
                epochs = 5
                print("Initiation complete")
                
                print("Training x-vector model")
                for e in range(epochs):
                    running_loss = 0.0
                    for i in range(int(nspkr)):
                        optimizer.zero_grad()
                        output=trainer(model(data[i]))
                        t1,_,t2=output.shape
                        #output=trainer(model(data[i]))
                        labels=i*torch.ones(t1,t2,dtype=torch.long)
                        loss=criterion(output,labels)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                        print("Spkeaker",i,"processed in epoch",e+1)
                    print('epoch: %d/%d calculated with loss: %.7f' %(e + 1,epochs, running_loss / nspkr))
                x=4
            else:
                print("No data to process")
        except:
            print("Training failed")
            
        try:
            if x==4:
                print("Finished Training, saving model!")
                torch.save(model.state_dict(), weigthFilename)
                print("X-vector model training complete!")

                    
            else:
                print("No model to save")
        except:
            print("Saving failed")
    except:
        print("No luck")

