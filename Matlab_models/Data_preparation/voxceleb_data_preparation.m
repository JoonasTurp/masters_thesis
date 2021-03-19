function voxceleb_data_preparation(inputdatadir,outputdatafile,nceps,noise_percentage)
%cepstrals calculated by following example:
%http://research.spa.aalto.fi/speech/s895150/ex1.html
% noise model in https://usir.salford.ac.uk/id/eprint/38902/1/571-CE020.pdf
try
    if ischar(noise_percentage), noise_percentage = str2double(noise_percentage); end
    if ischar(nceps), nceps = str2double(nceps); end
    
    % Step 1: Load data as audioDatastore
    addpath(genpath('/scratch/work/turpeim1/matlab/Mobirox_functions'));
    
    filename=cell(length(dir(sprintf('%s*.mat',inputdatadir))),1);
    
    ind=1;
    arraydir=dir(sprintf('%s/*.mat',inputdatadir));
    for ix=1:length(arraydir)
        file1=sprintf('%s/%s',inputdatadir,arraydir(ix).name);
        filename{ind}=file1;
        ind=ind+1;
    end
    
    %dataset_labels=cell(nspeakers,1);
    %data=cell(nspeakers,1);
    data={};
    dataset_labels={};
    speechlen=1024; %in this case, usually:audioSource.SamplesPerFrame;
    fs=16000;
    VAD_cst_param = vadInit_double_parpool(fs);
    %id10983/K3VF9KATPqc/00029 /home/jturpeinen/voxceleb/v1/voxceleb1_wav/id10983/K3VF9KATPqc_00029.wav 6.760063
    
    D = dctmtx(nceps);
    mel_weigths=fft2melmx(257, fs, nceps, 1, 50, 6855.5, 0);
    NR=100-10*(noise_percentage-1);
    if NR<100
        musanfile=matfile('musan_data.mat');
        musan=cell(3,1);
        musan{1}=musanfile.speech;
        musan{2}=musanfile.noise;
        musan{3}=musanfile.music;
        
        noise_ind=ones(3,1);
        file_ind=ones(3,1);
    end
catch error
    getReport(error)
    disp('Something went wrong with initialization');
    exit(0)
end


for fileind=1:length(filename)
    inputmat=matfile(filename{fileind});
    input_data=inputmat.audio_data;
    labels=inputmat.labels;

    speakers=unique(labels,'rows');
    nspk_file=size(speakers,1);
    for spk=1:nspk_file
        tic
        ind=zeros(length(labels),1);
        for i=1:length(labels)
            ind(i)=floor(mean((speakers(spk,:)==labels(i,:))));
        end
        
        utt_ind=find(ind==1);
        audio3=cell(length(utt_ind),1);
        %fprintf('Prosessing speaker %s with data size of [%d,%d]',labels{spk},size(input_data{spk},1),size(input_data{spk},2))
        for utt=1:length(utt_ind)
            try
                audio=input_data{utt_ind(utt)}(:);
                audio=double(audio)/max(abs(double(audio))); %normalize audio
                len=length(audio);
                % incase of non-unique rows in file with durations of original files:
                
                audio2=[];
                ind=1;
                for m=1:floor(len/speechlen)
                    s=1+(m-1)*speechlen;
                    e=m*speechlen;
                    
                    % triton issue??? this is not working
                    %vad=vadG729_double_mex(audio(s:e), VAD_cst_param);
                    %did not work on matlab/r2018a
                    
                    if vadG729_double_mex(audio(s:e), VAD_cst_param)==1
                        s2=1+(ind-1)*speechlen;
                        e2=ind*speechlen;
                        audio2(s2:e2)=audio(s:e);
                        ind=ind+1;
                    end
                end
                clear audio c
                noise_len=length(audio2);
                
                %type=[{'music'},{'speech'},{'noise'}];
                %temp_ceps=mel_weigths*abs(fft(audio(s:e)));
                %short-time Fourier spectrum as FFT(?):
                if NR<100
                    k=randperm(4,1); % select noise randomly from whitenoise or musan corpus
                    if k==1
                        % Noise is white noise
                        volume=integratedLoudness(audio2(:),fs);
                        noise=wgn(noise_len,1,volume);
                        noise=double(noise(:))/double(max(abs(noise)));
                        audio_noise=audio2(:)*NR+noise*(100-NR);
                    else
                        noise=[];
                        while length(noise)<noise_len
                            if file_ind(k-1)>length(musan{k-1}{noise_ind(k-1)}) && noise_ind(k-1)<length(musan{k-1})
                                file_ind(k-1)=1;
                                noise_ind(k-1)=noise_ind(k-1)+1;
                            elseif noise_ind(k-1)>=length(musan{k-1})
                                still_need=noise_len-length(noise);
                                noise=noise(:);
                                noise=repmat(noise,int32(1+floor(still_need/length(noise))),1);
                                file_ind(k-1)=1;
                                noise_ind(k-1)=1;
                            else
                                noise_audio=musan{k-1}{noise_ind(k-1)}{file_ind(k-1)};
                                noise(end+1:end+length(noise_audio))=noise_audio;
                                noise=noise(:);
                                file_ind(k-1)=file_ind(k-1)+1;
                            end
                            
                        end
                        noise=noise(1:noise_len);
                        noise=double(noise(:))/double(max(abs(noise)));
                        audio_noise=audio2(:)*NR+noise*(100-NR);
                    end
                    
                    audio_noise=audio_noise/100; % the overall audio should be scaled back to [-1,1]
                    clear noise
                else
                    audio_noise=audio2(:);
                end
                clear audio2
                stFs=spectrogram(filter([1 -0.97], 1, audio_noise), hamming(400), 240);
                
                
                % normalizing MFCC:
                ceps=log(mel_weigths*sqrt(abs(stFs))+1);
                ceps=abs(ceps);
                %m=1:size(ceps,2);m=0*m(:);v=m;
                temp=0*ceps;
                for j=1:size(ceps,2)
                    m=sum(ceps(:,j))/nceps;
                    v=sum((ceps(:,j)-m).^2)/(nceps-1);
                    temp(:,j)=(ceps(:,j)-m)/sqrt(v);
                end
                ceps=D*temp;
                ceps=clean_ceps(ceps);
                audio3{utt}=ceps;
                clear ceps
                
            catch error
                getReport(error)
                %fprintf('Error: Audio reading failed for audio (spk,utt): [%d, %d]\n',spk,utt)
                continue
            end
        end
        old_speaker_inds=zeros(length(dataset_labels),1);
        for i=1:length(dataset_labels)
            old_speaker_inds(i)=floor(mean((speakers(spk,:)==dataset_labels{i,1})));
        end
        old_spk=find(old_speaker_inds==1);
        if isempty(old_spk)==1
            data(end+1,1)={audio3};
            dataset_labels(end+1,1)={speakers(spk,:)};
        else
            data(old_spk,1)={[data{old_spk,1}; audio3]};
        end
        clear audio3
        toc
    end
    fprintf('Audiofile %s processed\n',filename{fileind})
end

labels=dataset_labels;

if isfile(outputdatafile)
    system(sprintf('rm %s',outputdatafile));
end
save(outputdatafile,'data','labels','-v7.3');

end

function VAD_cst_param = vadInit_double_parpool(fs)
% Algorithm Constants Initialization
%VAD_cst_param.Fs = double(8000);  % Sampling Frequency
VAD_cst_param.Fs=double(fs);
VAD_cst_param.M  = double(10);    % Order of LP filter
%VAD_cst_param.M=double(25); %ms
VAD_cst_param.NP = double(12);    % Increased LPC order
%VAD_cst_param.NP=double(25);
VAD_cst_param.No = double(128);   % Number of frames for long-term minimum energy calculation
VAD_cst_param.Ni = double(32);    % Number of frames for initialization of running averages
VAD_cst_param.INIT_COUNT = double(20);
% High Pass Filter that is used to preprocess the signal applied to the VAD
VAD_cst_param.HPF_sos = double([0.92727435, -1.8544941, 0.92727435, 1, -1.9059465, 0.91140240]);
%VAD_cst_param.L_WINDOW = double(240);  % Window size in LP analysis
VAD_cst_param.L_WINDOW=double(1024);%old:105+160;
VAD_cst_param.L_NEXT   = double(40);   % Lookahead in LP analysis
%VAD_cst_param.L_FRAME  = double(80);   % Frame size
VAD_cst_param.L_FRAME=double(265);
L1 = VAD_cst_param.L_NEXT;
L2 = VAD_cst_param.L_WINDOW - VAD_cst_param.L_NEXT;
VAD_cst_param.hamwindow = double([0.54 - 0.46*cos(2*pi*(0:L2-1)'/(2*L2-1));
    cos(2*pi*(0:L1-1)'/(4*L1-1))]);
% LP analysis, lag window applied to autocorrelation coefficients
VAD_cst_param.lagwindow = double([1.0001; exp(-1/2 * ((2 * pi * 60 / VAD_cst_param.Fs) * (1:VAD_cst_param.NP)').^2)]);
% Correlation for a lowpass filter (3 dB point on the power spectrum is
% at about 2 kHz)
VAD_cst_param.lbf_corr = double([0.24017939691329, 0.21398822343783, 0.14767692339633, ...
    0.07018811903116, 0.00980856433051,-0.02015934721195, ...
    -0.02388269958005,-0.01480076155002,-0.00503292155509, ...
    0.00012141366508, 0.00119354245231, 0.00065908718613, ...
    0.00015015782285]');
%VAD_cst_param.hamwindow=double(1024); %need 240*1 for speech_buf, else this double(ones(fs/100,1)) ok;
%this works: VAD_cst_param.lagwindow=double(ones(26,1));
%VAD_cst_param.lbf_corr=double(ones(26,1));
end

function ceps=clean_ceps(ceps)
if sum(sum(isnan(ceps)))~=0
    %ceps(n,:)=zeros(size(mel_weigths,1),1);
    [~,y]=find(isnan(ceps));
    y_nan=y(1);
    nan_ind=1;
    for ix=2:length(y)
        if y_nan(nan_ind)~=y(ix)
            nan_ind=nan_ind+1;
            y_nan(nan_ind)=y(ix);
        end
    end
    c=ceps(:,1:(y_nan(1)-1));
    for ix=2:length(y_nan)
        c=[c ceps(:,y_nan(ix-1)+1:y_nan(ix)-1)];
    end
    c=[c ceps(:,(y_nan(end)+1):end)];
    ceps=c;
    clear c
end
if sum(sum(isinf(ceps)))~=0
    [~,y]=find(isinf(ceps));
    y_inf=y(1);
    inf_ind=1;
    for ix=2:length(y)
        if y_inf(inf_ind)~=y(ix)
            inf_ind=inf_ind+1;
            y_inf(inf_ind)=y(ix);
        end
    end
    c=ceps(:,1:(y_inf(1)-1));
    for ix=2:length(y_inf)
        c=[c ceps(:,y_inf(ix-1)+1:y_inf(ix)-1)];
    end
    c=[c ceps(:,(y_inf(end)+1):end)];
    ceps=c;
    clear c
end

end
