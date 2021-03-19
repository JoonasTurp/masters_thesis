function gmm_ubm_speakers(datafilename,ubmFilename,modelFilename, dataset)
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

    
    if ~iscell(datafilename)        
        train_data=load_data(datafilename, dataset);
        nspkr=length(train_data);
    end
    %data_save=matfile(datafile,'Writable', true);
    %train_data=data_save.train_data;
    if ~isstruct(ubmFilename)
        ubm_file=matfile(ubmFilename, 'Writable', true);
        ubms=ubm_file.ubm;
    end
    
    tau = 10.0;
    lenData=length(train_data);
    
    
    speaker_models=cell(lenData,1);
    for u=1:length(ubms)
        if isempty(ubms{u}) && u<=length(ubms)
            ubm=ubms{u-1};
        elseif ~isempty(ubms{u}) && u==length(ubms)
            ubm=ubms{u};
        end
    end
    for spk=1:lenData
        tim = tic;
        nchannel=length(train_data{spk});
        N = 0; F = 0; S = 0;
        for cha=1:nchannel
            [n, f, s, ~] = expectation(train_data{spk}{cha}, ubm);
            N = N+n; F = F+f; S = S+s;
        end
        alpha = N ./ (N + tau); % trade-off between ML mean and UBM mean
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
        
        speaker_models{u,spk}=gmm;
        tim = toc(tim);
        fprintf('Speaker nro %d processed in %4.2f s\n',spk,tim)
    end

    save(modelFilename, 'speaker_models','-v7.3');
    
catch error
    getReport(error)
    disp('Error occured');
    exit(0)
end


end

function data = load_data(dataList, dataset)
% load all data into memory
datafile=matfile(dataList);
if length(dataset)==3 && mean(dataset=='dev')==1
    fileinfo=whos('-file', dataList);
    data=cell(fileinfo(1).size(1),1);
    for i=1:fileinfo(1).size(1)
        data{i}=datafile.data;
        l=floor(length(data{i})*0.8)+1;
        data{i}=data{i}(l:end,:);
    end
    
else
    data=datafile.data;
end
end

function [N, F, S, llk] = expectation(data, gmm)
% compute the sufficient statistics
mu=gmm.mu; sigma=gmm.sigma; w=gmm.w(:);
ndim = size(data, 1);
C = sum(mu.*mu./sigma) + sum(log(sigma));
D = (1./sigma)' * (data .* data) - 2 * (mu./sigma)' * data  + ndim * log(2 * pi);
post = -0.5 * (bsxfun(@plus, C',  D));
post = bsxfun(@plus, post, log(w));
xmax = max(post, [], 1);
llk    = xmax + log(sum(exp(bsxfun(@minus, post, xmax)), 1));
ind  = find(~isfinite(xmax));
if ~isempty(ind)
    llk(ind) = xmax(ind);
end

post = exp(bsxfun(@minus, post, llk));
N = sum(post, 2)';
F = data * post';
S = (data .* data) * post';
end