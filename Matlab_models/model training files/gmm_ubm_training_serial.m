function gmm_ubm_training_serial(starting_section,nmix,nceps, final_niter, ds_factor,nspeaker,dataDir, test_len)
try
    % based on gmm_em-function
    %{
    Inputs
        starting_section = If training was interrupted, from wich section
        the programs begins
        nmix             = How many mixture models for Gaussian mixture
        model
        nceps            = The number of cepstral coefficients in MFCC
        final_niter      = How many times 
        ds_factor        = How many frames are skipped for training the
        model during first iterations (no skipping during last 2
        iterations)
        nspeaker         = 0 for processing full data in dataDir, >0, for
        some other amount of speakers
        dataDir          = Location of data
        test_len         = Length of data for testing phase
    Outputs
        Update/generate GMM-UBM speaker recognition model from data
    
    
    %}
    
    % Step 1: Load data as audioDatastore
    addpath(genpath('/scratch/work/turpeim1/matlab'));
    ads = datastore(dataDir,'Type','file','ReadFcn',@audioread);

    if nspeaker==0 || nspeaker>length(ads.Files)
        lenData=length(ads.Files);
        
        testfile='/scratch/work/turpeim1/matlab/data/test_data-june.mat';
        trainfile='/scratch/work/turpeim1/matlab/data/train_data-june.mat';
        initial_ubmFilename='/scratch/work/turpeim1/matlab/data/init_ubm_gmm-june.mat';
        ubmFilename='/scratch/work/turpeim1/matlab/data/ubm_gmm-june.mat';
        modelFilename='/scratch/work/turpeim1/matlab/data/ubm_gmm_models-june.mat';
    else
        lenData=nspeaker;
        
        testfile=sprintf('/scratch/work/turpeim1/matlab/data/test_data-june-%d.mat',nspeaker);
        trainfile=sprintf('/scratch/work/turpeim1/matlab/data/train_data-june-%d.mat',nspeaker);
        initial_ubmFilename=sprintf('/scratch/work/turpeim1/matlab/data/init_ubm_gmm-june-%d.mat',nspeaker);
        ubmFilename=sprintf('/scratch/work/turpeim1/matlab/data/ubm_gmm-june-%d.mat',nspeaker);
        modelFilename=sprintf('/scratch/work/turpeim1/matlab/data/ubm_gmm_models-june-%d.mat',nspeaker);
    end

    section=starting_section;
    
    if section==1
        
        mu=zeros(nceps,1);
        sigma=zeros(nceps,1);
        nframes=0;
        % Step 2: Calculate features from audio
        train_data=cell(lenData,1);
        test_data=cell(lenData,1);
        
        for i=1:lenData
            tic
            input_audio=char(ads.Files(i));
            fprintf('\nProcessing audiofile %s\n',input_audio)
            if i==1
                %audioSource=dsp.AudioFileReader(input_audio);
                %fs=audioSource.SampleRate;
                [audio,fs]=audioread(input_audio);
                VAD_cst_param = vadInit_double_parpool(fs);
                speechlen=1024; %in this case, usually:audioSource.SamplesPerFrame;
                %clear audioSource
                mel_weigths=fft2melmx(speechlen, fs, nceps, 1, 133.33, 6855.5, 0);
            else
                audio=audioread(input_audio);
            end
            
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
            m=size(ceps,1);
            train_data{i}=ceps(1:floor(0.8*m),:)';
            test_data{i}=ceps(floor(0.8*m+1):end,:)';
            
            mu=mu+sum(train_data{i},2);
            nframes=nframes+size(train_data{1},2);
            fprintf('Audiofile %s processed\n',input_audio)
            clear ceps 
            toc
        end
        
        %save train and test data into file
        save(trainfile,'train_data','-v7.3');
        save(testfile,'test_data','-v7.3');
        clear test_data
        
        % Step 2.5 Initialize ubm-model (based on means and variance in data)
        fprintf('\nCalculating the GMM-UBM hyperparameters: sigma and mu ...\n');
        %average feature:
        mu=mu/nframes;
        %feature variance:
        for i=1:lenData
            for j=1:size(train_data{i},2)
                sigma=sigma+(train_data{i}(:,j)-mu).^2;
            end
        end
        sigma=sigma/(nframes-1);
        
        w=1;
        ubm.sigma=sigma;ubm.mu=mu;ubm.w=w;
        fprintf('\nSaving ubm to file %s\n', initial_ubmFilename);
        save(initial_ubmFilename,'ubm','-v7.3');
        fprintf('\n\nGMM hyperparameters initialized...\n');
        
        section=2;
    end
    
    if section==2
        if section==starting_section
            train_save=matfile(trainfile,'Writable', true);
            train_data=train_save.train_data;
            initial_ubm_file=matfile(initial_ubmFilename, 'Writable', true);
            ubm=initial_ubm_file.ubm;
            mu=ubm.mu;sigma=ubm.sigma;w=ubm.w(:);
        end
        
        % Step 3: Calculate ubm-UBM model (remember to add labels)
        if ischar(nmix), nmix = str2double(nmix); end
        if ischar(final_niter), final_niter = str2double(final_niter); end
        if ischar(ds_factor), ds_factor = str2double(ds_factor); end
        
        [ispow2, ~] = log2(nmix);
        if ( ispow2 ~= 0.5 )
            % default nmix=256
            nmix=2^8;
        end
        
        niter = [1 2 4 4  4  4  6  6   10  10  15];
        niter(log2(nmix) + 1) = final_niter;
        
        mix=1;
        while ( mix <= nmix )
            if ( mix >= nmix/2 ), ds_factor = 1; end % not for the last two splits!
            fprintf('\nRe-estimating the ubm hyperparameters for %d components ...\n', mix);
            for iter = 1 : niter(log2(mix) + 1)
                fprintf('EM iter#: %d \t', iter);
                N = 0; F = 0; S = 0; L = 0; nframes = 0;
                w=w(:);
                tim = tic;
                for ix = 1 : lenData
                    data_patch=train_data{ix}(:, 1:ds_factor:end);
                    n = size(data_patch, 1);
                    C = sum(mu.*mu./sigma) + sum(log(sigma));
                    D = (1./sigma)' * (data_patch .* data_patch) - 2 * (mu./sigma)' * data_patch  + n * log(2 * pi);
                    post = -0.5 * (bsxfun(@plus, C',  D));
                    post = bsxfun(@plus, post, log(w));
                    
                    dim=1;
                    xmax = max(post, [], dim);
                    l    = xmax + log(sum(exp(bsxfun(@minus, post, xmax)), dim));
                    ind  = find(~isfinite(xmax));
                    if ~isempty(ind)
                        l(ind) = xmax(ind);
                    end
                    
                    post = exp(bsxfun(@minus, post, l));
                    
                    N = N + sum(post, 2)';
                    F = F + data_patch * post';
                    S = S + (data_patch .* data_patch) * post';
                    L = L + sum(l);
                    nframes = nframes + length(l);
                end
                tim = toc(tim);
                fprintf('[llk = %.2f] \t [elaps = %.2f s]\n', L/nframes, tim);
                
                
                w  = N / sum(N);
                mu = bsxfun(@rdivide, F, N);
                sigma = bsxfun(@rdivide, S, N) - (mu .* mu);
                floor_const=0.1;
                % set a floor on covariances based on a weighted average of component
                % variances
                vFloor = sigma * w' * floor_const;
                sigma  = bsxfun(@max, sigma, vFloor);
                % sigma = bsxfun(@plus, sigma, 1e-6 * ones(size(sigma, 1), 1));
            end
            
            if ( mix < nmix )
                [n, m] = size(sigma);
                [sig_max, arg_max] = max(sigma);
                eps = sparse(0 * mu);
                eps(sub2ind([n, m], arg_max, 1 : m)) = sqrt(sig_max);
                % only perturb means associated with the max std along each
                % dimension
                mu = [mu - eps, mu + eps];
                % mu = [mu - 0.2 * eps, mu + 0.2 * eps]; % HTK style
                sigma = [sigma, sigma];
                w = [w, w] * 0.5;
            end
            mix = mix * 2;
        end
        
        ubm.w=w(:);ubm.sigma=sigma;ubm.mu=mu;
        save(ubmFilename,'ubm','-v7.3');
        section=3;
    end
    
    if section==3
        if section==starting_section
            train_save=matfile(trainfile,'Writable', true);
            train_data=train_save.train_data;
            ubm_file=matfile(ubmFilename, 'Writable', true);
            ubm=ubm_file.ubm;
        end
        tau = 10.0;
        speaker_models = cell(lenData, 1);
        for s=1:lenData
            tim = tic;

            [N, F, S, ~] = gmm_em_expectation_mex(train_data{s}, ubm.mu, ubm.sigma, ubm.w(:));
            
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
            
            speaker_models{s}=gmm;
            tim = toc(tim);
            fprintf('Speaker nro %d processed in %4.2f s\n',s,tim)
        end
        
        save(modelFilename, 'speaker_models','-v7.3');
        section=4;
    end
    
    if section==4
        plotting=0;
        disp('Evaluating speaker models');
        gmm_ubm_result4(test_len, ubmFilename, modelFilename,testfile,lenData,plotting)
    end
    
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