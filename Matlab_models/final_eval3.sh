#!/bin/bash -l

#SBATCH --time=15:00:00
#SBATCH --mem-per-cpu=60G
#SBATCH --array=1-8
#SBATCH -o /scratch/work/turpeim1/matlab/log/final-eval-%A_%a.out
##SBATCH -o /scratch/work/turpeim1/matlab/log/final-eval-%j.out

module load matlab/r2019a

########noise_percentage=1
noise_percentage=$SLURM_ARRAY_TASK_ID



nceps=20;
ubmFilename=/scratch/work/turpeim1/matlab/data/ubm_gmm-dev-full-$nceps-$noise_percentage-full.mat;


############################################################
for noise_percentage2 in 1 2 3 4 5 6 7 8
do
if (( $noise_percentage != $noise_percentage2 ));
then
for dataset in voxceleb1_wav_test test atr_kids
do

feature_datafile=/scratch/work/turpeim1/matlab/data/$dataset-full-nceps-$nceps-noise_level-$noise_percentage2.mat;
if [ -f "$feature_datafile" ]; then
echo "file exists $feature_datafile"

else
if [ $dataset != atr_kids ];then

inputdatadir=/scratch/work/turpeim1/voxceleb/$dataset/
srun matlab -nojvm -nosplash -r "voxceleb_data_preparation5('$inputdatadir','$feature_datafile',$nceps, $noise_percentage2); exit(0)"

else

srun matlab -nojvm -nosplash -r "child_data_preparation('$feature_datafile',$nceps, $noise_percentage2, 0); exit(0)"

fi

fi


echo "Evaluating ubm for $dataset"

modelFilename=/scratch/work/turpeim1/matlab/data/gmm_ubm_models-$dataset-full-nceps-$nceps-datanoise_level-$noise_percentage2-modelnoiselevel-$noise_percentage.mat;
scoreFile=/scratch/work/turpeim1/matlab/data/gmm-ubm-scores-$dataset-full-nceps-$nceps-datanoise_level-$noise_percentage2-modelnoiselevel-$noise_percentage.mat;

test_length=400
nspkr=0
just_plot=0;
if [ -f "$modelFilename" ]; then
echo "Modelfile exists"
fi
if [ -f "$scoreFile" ]; then
echo "scoreFile exists"
fi

srun matlab -nojvm -nosplash -r "gmm_ubm_speakers('$feature_datafile','$ubmFilename','$modelFilename', '$dataset'); exit(0)"
srun matlab -nojvm -nosplash -r "ubm_results('$test_length', '$nspkr','$ubmFilename','$modelFilename','$feature_datafile','$scoreFile', 0); exit(0)"

done
else
echo "Already tested during jobid 48972813"
fi
done
echo "Recipe tested by noise_ind $noise_percentage"
