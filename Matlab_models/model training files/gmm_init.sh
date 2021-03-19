#!/bin/bash -l

#SBATCH --time=24:00:00
#SBATCH --mem=8G    ## datan läpikäyntiin kulu 7733M muistia eli noin 8G riittää
#SBATCH -o /scratch/work/turpeim1/matlab/log/gmm_ubm_init-%j.out

module load matlab/r2019a

nmix=256
final_niter=15
nspeaker=0;
export OMP_NUM_THREADS=$(nproc)

starting_section=4;
nceps=40;
ds_factor=10;
test_len=200; #6.4*2s=12.8s
srun matlab -nojvm -nosplash -r "gmm_ubm_training_serial($starting_section, $nmix, $nceps, $final_niter, $ds_factor,$nspeaker,$test_len); exit(0)"