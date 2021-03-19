function gmm_ubm_update_speaker_model(sample,fs,speechlen,studentID,nceps,ubmFilename,modelFilename,trainfile)
try
    %{
    
    inputs
        sample          = doubleprecision vector from recorded audiosample
        fs              = samplerate of audio
        speechlen       = length of frame to be read (samples per frame)
        studentID       = zero (0), if recording belongs to new student in class,
                        or nonzero and positive integer when old model is updated
        nceps           = number of cepstral coefficients calclulated from audioframe
        ubmFilename     = matfile where universal backround model is located
        modelFilename   = matfile, where speaker model is located or where it
                        will be updated
        trainfile       = matfile, where speaker training data is located or where 
                        it will be updated
    
     outputs
        updates classmodel with a new studentmodel, or old model
        studentmodel with new speechsample
    %}
    if ischar(studentID) || isstring(studentID)
        %studentID is not number
        studentID=str2double(studentID);
    elseif ~isa(studentID,'double')
        %studentID is not double precision
        studentID=double(studentID);
    end
    %if at this point, the studentID is not number despite conversion, we
    %have a serious error:
    if ~isa(studentID,'double')
        disp('StudentID must be s number');
        exit(0)
    end
    % based on gmm_em-function
    addpath(genpath('/scratch/work/turpeim1/matlab'));
    tic
    
    ubm_file=matfile(ubmFilename, 'Writable', true);
    ubm=ubm_file.ubm;
        
    tau = 10.0;
    
    VAD_cst_param = vadInit_double_parpool(fs);
    mel_weigths=fft2melmx(speechlen, fs, nceps, 1, 133.33, 6855.5, 0);
    
    len=length(sample);
    if mod(len,speechlen)~=0
        old_end=len;
        new_end=len+speechlen-mod(len,speechlen);
        sample(old_end+1:new_end)=0;
    end
    ceps=zeros(floor(new_end/speechlen),size(mel_weigths,1));
    n=1;
    for m=1:(new_end/speechlen)
        s=1+(m-1)*speechlen;
        e=m*speechlen;
        % triton issue??? this is not working
        vad=vadG729_double_mex(sample(s:e), VAD_cst_param);
        %did not work on matlab/r2018a
        
        if vad==1
            temp_ceps=mel_weigths*abs(fft(sample(s:e)));
            if sum(isnan(temp_ceps))==0
                ceps(n,:)=temp_ceps;
            else
                ceps(n,:)=zeros(size(mel_weigths,1),1);
            end
            n=n+1;
        end
    end
    
    clear sample
    ceps=ceps';
    
    trainingFile=matfile(trainfile, 'Writable', true);
    class_File=matfile(modelFilename, 'Writable', true);
    if studentID==0
        lenData = size(class_File,'speaker_models');
        s=lenData+1;
        trainingFile.train_data(s,1)={ceps};
        speech_sample=ceps;
    else
        s=studentID;
        speech_sample=trainingFile.train_data(s,1);
        speech_sample=[speech_sample{:},ceps];
        trainingFile.train_data(s,1)={speech_sample};

    end

    [N, F, S, ~] = gmm_em_expectation_mex(speech_sample, ubm.mu, ubm.sigma, ubm.w(:));
    alpha = N ./ (N + tau); % tarde-off between ML mean and UBM mean
    m_ML = bsxfun(@rdivide, F, N);
    m = bsxfun(@times, ubm.mu, (1 - alpha)) + bsxfun(@times, m_ML, alpha);
    gmm.mu = m;
    
    alpha = N ./ (N + tau);
    v_ML = bsxfun(@rdivide, S, N);
    v = bsxfun(@times, (ubm.sigma+ubm.mu.^2), (1 - alpha)) + bsxfun(@times, v_ML, alpha) - (m .* m);
    gmm.sigma = v;
    
    
    alpha = N ./ (N + tau);
    w_ML = N / sum(N);
    w = bsxfun(@times, ubm.w, (1 - alpha)) + bsxfun(@times, w_ML, alpha);
    w = w / sum(w);
    gmm.w = w;
    
    class_File.speaker_models(s,1)={gmm};
    fprintf('\nSpeaker nro %d processed in ',s)
    toc
    fprintf(' seconds\n')
    save(modelFilename, 'speaker_models','-v7.3');


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