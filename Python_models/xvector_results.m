function xvector_results(scoreFilename, xvectordir, train_plda, pldafile, plotting)


disp('Evaluating speaker models');
addpath(genpath('/scratch/work/turpeim1/matlab'));
addpath(genpath('/scratch/work/turpeim1/matlab/msr/code'));
if ischar(plotting), plotting = str2double(plotting); end
plotdir="/scratch/work/turpeim1/python_models/plots/";


if iscell(scoreFilename)
    scores=scoreFilename;
    clear scoreFilename
else
    scorefile=matfile(scoreFilename);
    %scores=scorefile.score;

    %answers=scorefile.answers;
    %if iscell(scores)
    %    scores=scores{:};
    %end
    %if iscell(answers)
    %    answers=answers{:};
    %end
end

if plotting==1
    scores=scorefile.scores;
    answers=scorefile.answers;
    ivScores=scorefile.ivScores
    nSpeakers=size(scores,1);
    % Step4: Now compute the EER and plot the DET curve and confusion matrix
    
    %nspk=length(score.gmmScores(:,1));
    figure(1)
    imagesc(scores);
    title('Speaker Verification Likelihood (GMM Model)');
    ylabel('Test # (Channel x Speaker)'); xlabel('Model #');
    colorbar; drawnow; axis xy

    figure(2)
    %eer = compute_eer(p, answers, true)
    [eer, dcf08, dcf10] = compute_eer(reshape(scores,size(scores1,1)*size(scores1,2),1), reshape(answers,size(answers,1)*size(answers,2),1), 1)
    
    figure(3)
    imagesc(scores);
    title('Speaker Verification Likelihood (GMM Model)');
    ylabel('Test # (Channel x Speaker)'); xlabel('Model #');
    colorbar; drawnow; axis xy

    figure(4)
    %eer = compute_eer(p, answers, true)
    [eer_plda, dcf08_plda, dcf10_plda] = compute_eer(reshape(ivScores, size(ivScores,1)*size(ivScores,2), 1), reshape(answers, size(answers,1)*size(answers,2), 1), 1)
    saveas(gcf,'plots/chart.jpg')
    
else
    scoredir=dir(sprintf('%s*',xvectordir));
    xvectors=[];
    labels=[];
    for file=1:length(scoredir)
            scorefile=matfile(sprintf('%s/%s',scoredir(file).folder, scoredir(file).name));
            xvectors=[xvectors; scorefile.xvectors];
            labels=[labels, scorefile.labels];
    end
    labels=labels(:);
    
    [scores, answers, models]=calc_scores(xvectors,labels);
    
    [eer, dcf08, dcf10] = compute_eer(reshape(scores,size(scores,1)*size(scores,2),1), reshape(answers,size(answers,1)*size(answers,2),1), 0)
    correct=0;
    for ix =1:size(scores,1)
        [~, argmax] = max(scores(ix,:));
        
        correct=correct+answers(ix,argmax);
        %correct=correct+(labels(argmax)==labels(ix));
    end
    accuracy=correct/size(scores,1)*100;
    fprintf('\nAccuracy without plda is %f percent\n',accuracy)
    if train_plda==1
        fprintf('\nNow training plda \n')
        speakers=unique(labels);
        plda = gplda_em(double(xvectors'), labels(:), min(100, length(speakers)-1), 10);
        save(pldafile,'plda')
    else
        pldamat=matfile(pldafile);
        plda=pldamat.plda;
    end
    ivScores = score_gplda_trials(plda, models, xvectors);
    if mean(size(ivScores)==size(scores))~=1
        ivScores=ivScores';
    end
    save(scoreFilename,'scores','answers','models','ivScores','-v7.3')
    [eer_plda, dcf08_plda, dcf10_plda] = compute_eer(reshape(ivScores, size(ivScores,1)*size(ivScores,2), 1), reshape(answers, size(answers,1)*size(answers,2), 1), 0)
    correct_plda=0;
    for ix =1:size(ivScores,1)
        [~, argmax] = max(ivScores(ix,:));
        
        correct_plda=correct_plda+answers(ix,argmax);
        %correct=correct+(labels(argmax)==labels(ix));
    end
    accuracy_plda=correct_plda/size(ivScores,1)*100;
    fprintf('\nAccuracy with plda is %f percent\n',accuracy_plda)
end

end

function [scores, answers,models]=calc_scores(xvectors,labels)
    speakers=unique(labels);
    models=zeros(length(speakers),size(xvectors,2));
    for spk =1:length(speakers)
        inds=find(labels==speakers(spk));
        models(spk,:)=mean(xvectors(inds,:),1);
    end
    scores=zeros(length(speakers),size(xvectors,1));
    answers=zeros(length(speakers),size(xvectors,1));
    for ix =1:length(speakers)
        for jx =1:size(xvectors,1)
            scores(ix,jx)=dot(models(ix,:), xvectors(jx,:))/(norm(models(ix,:))*norm(xvectors(jx,:)));
            if speakers(ix)==labels(jx)
                answers(ix,jx)=1;
            end
        end
    end
end
