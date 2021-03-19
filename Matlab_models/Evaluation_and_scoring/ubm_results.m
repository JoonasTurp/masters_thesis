function ubm_results(testlen, nSpeakers, ubmFilename, modelFilename, testFilename, scoreFilename, plotting)


disp('Evaluating speaker models');
addpath(genpath('/scratch/work/turpeim1/matlab'));
addpath(genpath('/scratch/work/turpeim1/matlab/new_functions'));
if ischar(plotting), plotting = str2double(plotting); end
if ischar(nSpeakers),nSpeakers = str2double(nSpeakers); end
if ischar(testlen), testlen = str2double(testlen); end


if plotting==0
    try
        if iscell(ubmFilename)
            ubm=ubmFilename;
            clear ubmFilename
        else
            ubm_file=matfile(ubmFilename);
            ubms=ubm_file.ubm;
            for u=1:length(ubms)
                if isempty(ubms{u}) && u<=length(ubms)
                    ubm=ubms{u-1};
                elseif ~isempty(ubms{u}) && u==length(ubms)
                    ubm=ubms{u};
                end
            end
        end
        
        if iscell(testFilename)
            test_data=testFilename;
            clear testFilename
        else
            testfile=matfile(testFilename);
            test_data=testfile.data;
        end

        if nSpeakers==0
            nSpeakers=length(test_data);
        end
        set_len=100000000000;
        for spk=1:nSpeakers
            set_len=min(set_len,length(test_data{spk}));
        end
        
        if iscell(modelFilename)
            models=modelFilename;
        else
            modelfile=matfile(modelFilename);
            models=modelfile.speaker_models;
            models_fixed={};
            for i=1:size(models,1)
                for j=1:size(models,2)
                    if ~isempty(models{i,j})
                        models_fixed(end+1,1)=models(i,j);
                    end
                end
            end
            clear models
            models=models_fixed;
        end
        gmmScores=zeros(set_len*nSpeakers,1);
        answers=zeros(set_len*nSpeakers,1);
        
    catch error
        getReport(error)
        disp('Error occured while loading data');
        exit(0)
    end
    try
        %actual evaluation
        correct=0;
        set_ind=1;
        tot_answers=0;
        for id=1:nSpeakers
            %nchannel=length(test_data{id});
            fprintf('\nScore calculations for speaker %d/%d',id, nSpeakers);
            for cha=1:set_len
                % speech data of speaker
                
                
                % are we testing the full data or just part of it?
                if testlen==0 || testlen>size(test_data{id}{cha},2)
                    fea=test_data{id}{cha};
                else
                    fea=test_data{id}{cha}(:,1:testlen);
                end
                
                mu=ubm.mu; sigma=ubm.sigma; w=ubm.w(:);
                
                ndim = size(fea, 1);
                C = sum(mu.*mu./sigma) + sum(log(sigma));
                D = (1./sigma)' * (fea .* fea) - 2 * (mu./sigma)' * fea  + ndim * log(2 * pi);
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
                llr = zeros(nSpeakers, 1);
                set_answers = zeros(nSpeakers, 1);
                for tr = 1 : nSpeakers
                    gmm = models{tr};    % speaker models we test
                    mu=gmm.mu; sigma=gmm.sigma; w=gmm.w(:);
                    ndim = size(fea, 1);
                    C = sum(mu.*mu./sigma) + sum(log(sigma));
                    D = (1./sigma)' * (fea .* fea) - 2 * (mu./sigma)' * fea  + ndim * log(2 * pi);
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
                    llr(tr) = mean(llk(:) - ubm_llk(:));
                    if id==tr
                        set_answers(tr)=1;
                    end
                end
                tot_answers=tot_answers+1;
                [~, argmax] = max(tr);
                if argmax==id
                   correct=correct+1; 
                end
                %gmmScores(end+1:end+nSpeakers,1)=llr;
                %answers(end+1:end+nSpeakers,1)=set_answers;
                gmmScores(set_ind:set_ind+nSpeakers-1,1)=llr;
                answers(set_ind:set_ind+nSpeakers-1,1)=set_answers;
                set_ind=set_ind+nSpeakers;
            end
            
        end
        
        %gmmScores=reshape(gmmScores,nspks, nspks);
        score=matfile(scoreFilename, 'Writable', true);
        score.gmmScores(1,1)= {gmmScores};
        score.answers(1,1)= {answers};
        fprintf('\nAccuracy of recognition is:\n');
        score.accuracy(1,1)=correct/tot_answers*100
        fprintf('\nDone with scoring speakers\n');
    catch error
        getReport(error)
        disp('Error occured while calculating scores');
        exit(0)
    end
elseif plotting==1 && isfile(scoreFilename)
    try
        testfile=matfile(testFilename);
        test_data=testfile.data;
        nSpeakers=length(test_data);
        l=zeros(nSpeakers,1);
        for i=1:length(test_data)
            l(i)=size(test_data{i},1);
        end
        clear testfile test_data
        score=matfile(scoreFilename);
        gmmScores=score.gmmScores;
        answers=score.answers;
        answers=answers{:};
        gmmScores=gmmScores{:};
        ind=find(gmmScores~=0);
        gmmScores=gmmScores(ind,1);
        answers=answers(ind,1);

        %figure(1)
        %imagesc(gmmScores);
        %title('Speaker Verification Likelihood (GMM Model)');
        %ylabel('Test # (Channel x Speaker)'); xlabel('Model #');
        %colorbar; drawnow; axis xy
        
        %figure(2)
        [eer, dcf08, dcf10] = new_compute_eer(gmmScores, answers, 1)

        set_ind=1;
        set_len=min(l);
        correct=0;
        tot=0;
        for i=1:nSpeakers
            for j=1:set_len
                llr=gmmScores(set_ind:set_ind+nSpeakers-1,1);
                set_ans=answers(set_ind:set_ind+nSpeakers-1,1);
                set_ind=set_ind+nSpeakers;
                [~, argmax] = max(llr);
                [~, argmax2] = max(set_ans);
                correct=correct+(argmax==argmax2);
                tot=tot+1;
            end
        end
        fprintf('\nAccuracy is ...\n');
        accuracy=correct/tot*100
    catch error
        getReport(error)
        disp('Error occured while verifying scores');
        exit(0)
    end
else
    try
        if plotting~=0 && plotting~=1
            disp('plotting must be either 0 or 1')
        end
        if ~isfile(scoreFilename)
            disp('No scorefile found')
        end
    catch error
        getReport(error)
        disp('Error occured due to bad programming');
        exit(0)
    end
end


end
