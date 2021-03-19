function id=gmm_ubm_whoisspeaking(input_audio,fs,speechlen,ubmFilename,modelFilename,nceps,testgroup)
try
    %{
     inputs
        input_audio   = location of audiofile where sample is located, or
                        numeric vector corresponding to audio sample
        fs            = samplerate of audio 
        speechlen     = length of frame to be read (samples per frame) 
        ubmFilename   = matfile where universal backround model is located
        modelFilename = matfile, where speaker models are located 
        nceps         = how many cepstral coefficients per frame 
        testgroup     = indexes of speakers in modelfile to be tested
    
     output
         id           = id listed in testgroup, whith highest correspondence
                        to speech sample (our guess for the speaker)
    %}
    % based on gmm_em-function
    % Step 1: Load data as audioDatastore
    addpath(genpath('/scratch/work/turpeim1/matlab'));
    
    %{
    testfile='/scratch/work/turpeim1/matlab/data/test_data-june.mat';
        trainfile='/scratch/work/turpeim1/matlab/data/train_data-june.mat';
        initial_ubmFilename='/scratch/work/turpeim1/matlab/data/init_ubm_gmm-june.mat';
        ubmFilename='/scratch/work/turpeim1/matlab/data/ubm_gmm-june.mat';
        modelFilename='/scratch/work/turpeim1/matlab/data/ubm_gmm_models-june.mat';
    datadir='/scratch/work/turpeim1/matlab/data/';
    nceps=40;
    dataDir = '/scratch/work/turpeim1/voxceleb/voxceleb1_wav/';
    ads = datastore(dataDir,'Type','file','ReadFcn',@audioread);
    input_audio=ads.Files{100};
    [audio,fs]=audioread(input_audio);
    input{1}={audio};input{2}=fs;
    audioSource=dsp.AudioFileReader(input_audio);
    input{3}=audioSource.SamplesPerFrame;
    testgroup=[1,2,3,4,5,50,60,70,80,90,100];
    %}
    
    modelfile=matfile(modelFilename);
    models=modelfile.speaker_models;
    ubs=matfile(ubmFilename);
    ubm=ubs.ubm;
    % Step 2: Calculate cepstures from audio
    if ischar(input_audio)
        [audio,fs]=audioread(input_audio);
        speechlen=1024; %in this case, usually:audioSource.SamplesPerFrame;
    elseif isnumeric(input_audio)
        audio=input_audio;
        %audio=audio{:};
        %fs=input_audio{2};
        %speechlen=double(input_audio{3});
    else
        disp('Insufficient information! What is the samplerate or the audio?')
        exit(0)
    end
    
    VAD_cst_param = vadInit_double_parpool(fs);
    mel_weigths=fft2melmx(speechlen, fs, nceps, 1, 133.33, 6855.5, 0);
    
    len=length(audio);
    if mod(len,speechlen)~=0
        old_end=len;
        new_end=len+speechlen-mod(len,speechlen);
        audio(old_end+1:new_end)=0;
    end
    
    ceps=zeros(floor(new_end/speechlen),size(mel_weigths,1));
    n=1;
    for m=1:(new_end/speechlen)
        s=1+(m-1)*speechlen;
        e=m*speechlen;
        % triton issue??? this is not working
        vad=vadG729_double_mex(audio(s:e), VAD_cst_param);
        %did not work on matlab/r2018a
        
        if vad==1
            temp_ceps=mel_weigths*abs(fft(audio(s:e)));
            if sum(isnan(temp_ceps))==0
                ceps(n,:)=temp_ceps;
            else
                ceps(n,:)=zeros(size(mel_weigths,1),1);
            end
            n=n+1;
        end
    end
    
    clear audio
    ceps=ceps';
    %speechfile=[datadir,'ubm_gmm-',datestr(now, 'HH:MM:SS_dd-mm-yyyy'),'.mat'];
    %save(speechfile,'ceps','-v7.3');
    fprintf('Audiofile was preprocessed\n')
    
    % compute model from unknown speaker
    mu=ubm.mu; sigma=ubm.sigma; w=ubm.w(:);
    ndim = size(ceps, 1);
    C = sum(mu.*mu./sigma) + sum(log(sigma));
    D = (1./sigma)' * (ceps .* ceps) - 2 * (mu./sigma)' * ceps  + ndim * log(2 * pi);
    post = -0.5 * (bsxfun(@plus, C',  D));
    post = bsxfun(@plus, post, log(w));
    
    % 2. compute log(sum(exp(x),dim)) while avoiding numerical underflow
    xmax = max(post, [], 1);
    llk    = xmax + log(sum(exp(bsxfun(@minus, post, xmax)), 1));
    ind  = find(~isfinite(xmax));
    if ~isempty(ind)
        llk(ind) = xmax(ind);
    end
    
    ubm_llk=llk;
    clear llk
    
    %Scoring unknown speaker model for known speaker models
    llr = zeros(length(testgroup), 1);
    for m = 1 : length(testgroup)
        tr=testgroup(m);
        gmm = models{tr};    % speaker models we test
        mu=gmm.mu; sigma=gmm.sigma; w=gmm.w(:);
        ndim = size(ceps, 1);
        C = sum(mu.*mu./sigma) + sum(log(sigma));
        D = (1./sigma)' * (ceps .* ceps) - 2 * (mu./sigma)' * ceps  + ndim * log(2 * pi);
        post = -0.5 * (bsxfun(@plus, C',  D));
        post = bsxfun(@plus, post, log(w));
        
        % 2. compute log(sum(exp(x),dim)) while avoiding numerical underflow
        xmax = max(post, [], 1);
        maybenans=exp(bsxfun(@minus, post, xmax));
        llk=zeros(length(xmax),1);
        for i=1:size(maybenans,2)
            xnan=find(~isnan(maybenans(:,i)));
            llk(i)=xmax(i)+log(sum(maybenans(xnan,i)));
        end
        %llk    = xmax + log(sum(nonans, 1));
        ind  = find(~isfinite(xmax));
        if ~isempty(ind)
            llk(ind) = xmax(ind);
        end
        llr(m) = mean(llk(:) - ubm_llk(:));
    end
    
    %best match is id
    id_in_testgroup=find(llr==max(llr));
    id=testgroup(id_in_testgroup);
    
catch error
    getReport(error)
    disp('Error occured');
    exit(0)
end

end


function VAD_cst_param = vadInit_double_parpool(fs)
%fs=16000;
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