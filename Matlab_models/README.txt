The process is as follows:
1) Process speech data from audiofiles to mel frequency cepstrals with:
 
2) Train recognition model with x_vector_trainer.py. 
3) Calculate x-vectors with xvector.py
4) Calculate scores with xvector_results.m. NOTE! This is a matlab file


NOTE!
In both Python and Matlab models, the file names are incorrect. You can use my *.sh files to see my naming convention

The Matlab model is trained with MSR Identity Toolbox v1.0:
https://www.microsoft.com/en-us/research/publication/msr-identity-toolbox-v1-0-a-matlab-toolbox-for-speaker-recognition-research-2/

Voice activity is evaluated with G.729 Voice Activity Detection:
https://www.mathworks.com/help/dsp/ug/g-729-voice-activity-detection.html;jsessionid=9bb808662356f30b0b7ceed69aa0

Cepstral Coefficients were calculated with:
fft2melmx from Rastamat package:
https://www.ee.columbia.edu/~dpwe/resources/matlab/rastamat/