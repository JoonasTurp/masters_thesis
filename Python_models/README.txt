The process is as follows:
1) Process data speech data into numpy array. 
For this you can use data_preprocess_shuffled_numpy_array.py
2) Train recognition model with x_vector_trainer.py. 
3) Calculate x-vectors with xvector.py
4) Calculate scores with xvector_results.m. NOTE! This is a matlab file


NOTE!
The model trained with xvectors is awful. I cannot recommend to use it for speaker recognition.
The matlab model is much more accurate. In both Python and Matlab models, the file names are incorrect.

The python model is based on Kaldi x-vector model developed by Chau Luu 
from University of Edinburgh. You can find her work from github: https://github.com/cvqluu