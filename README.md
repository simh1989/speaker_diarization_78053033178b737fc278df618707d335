# Speaker Diarization

### How to start?
1. create virtual environment using virtualenv (you need to activate it whenever you run the script)
2. install the required packages using `pip install -r requirementx.txt`

### What is included?
For an example, please check this [notebook](https://github.com/simh1989/speaker_diarization_78053033178b737fc278df618707d335/blob/master/demo_pyAudio.ipynb)
1. Initialize the instance for speakerDiarization class  
   ``` python
   from speakerDiarization import speakerDiarization as sD 
   sd = sD.speakerDiarization(audio_filepath='',
                              vad_filepath='',
                              num_speakers=2)
   ```
   
2. Calculate Speaker Diarization and return new VAD file  
   ``` python
   new_vad = sd.speaker_diarization(output_filepath='')
   ```
   
   There are some other parameters you can tune for the speaker diarization algorithm.  
   * sd.mid_window, mid-term window size
   * sd.mid_step, mid-term window step
   * sd.short_window, short-term window size
   * sd.lda_dim, LDA dimension (0 for no LDA)    
   
   Please refere to [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis/blob/4c7c2cfa068dcdd72427d106ae64f38d33f1570f/pyAudioAnalysis/audioSegmentation.py#L800) for more details.
   
3. Calculate the accuracy
   ``` python
   accuracy = sd.evaluate_diarization_accuracy(ground_truth_filepath)
   ```
   
   Accuracy = total lengh of correct classified time / total lenght of time
   
4. Calculate the average cluster and speaker [purity](https://stats.stackexchange.com/questions/95731/how-to-calculate-purity)
   ``` python
   average_cluster_purity, average_speaker_purity = sd.evaluate_diarization_purity(ground_truth_filepath)
   ```
   
5. Plot
   ``` python
   sd.plot_diarization()
   ```
