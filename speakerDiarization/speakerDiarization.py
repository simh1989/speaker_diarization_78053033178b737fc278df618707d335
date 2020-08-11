import json
import numpy as np
from matplotlib.pylab import plt
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioSegmentation as aS

class speakerDiarization:
    
    def __init__(self, audio_filepath, vad_filepath, num_speakers):
        """
        Initialization of speakerDiarization class
        
        Parameters
        ----------
        audio_filepath : str
            filepath to wav audio file
        vad_filepath : str
            filepath to vad file
        num_speakers : int
            number of speakers

        Returns
        -------
        None
        
        """
        self.audio_filepath = audio_filepath
        with open(vad_filepath, 'rb') as f:
            self.vad = json.load(f)
        self.num_speakers = num_speakers
        
        sampling_rate, signal = audioBasicIO.read_audio_file(audio_filepath)
        self.duration = len(signal) / sampling_rate
        
        self.rlt = None
        self.gt = None
            
        # Default parameters, this is from pyAudioAnalysis 
        self.mid_window=2.0
        self.mid_step=0.2
        self.short_window=0.05
        self.lda_dim=35
    
    def speaker_diarization(self, output_filepath='new_vad.json'):
        """
        This function run speaker diarization and return the results to original VAD input
        
        Parameters
        ----------
        output_filepath : str
            filepath to save the output vad file. (Default: new_vad.json)

        Returns
        -------
        new_vad : list
            list of {start_time, end_time, speaker}
        
        """
        self.rlt = aS.speaker_diarization(filename=self.audio_filepath,
                                     n_speakers=self.num_speakers,
                                     mid_window=self.mid_window,
                                     mid_step=self.mid_step,
                                     short_window=self.short_window,
                                     lda_dim=self.lda_dim)
        
        self.rlt_timestamp = np.array(range(len(self.rlt))) * self.mid_step
        
        # Remap the results to the vad time segments
        new_vad = []
        for each_vad in self.vad:
            start_time = each_vad['start_time']
            end_time = each_vad['end_time']

            vad_labels = [0] * self.num_speakers
            for i in range(len(self.rlt_timestamp)):
                each_t = self.rlt_timestamp[i]
                if each_t >= start_time and each_t < end_time:
                    if each_t + self.mid_step < end_time:
                        vad_labels[int(self.rlt[i])] = vad_labels[int(self.rlt[i])] + self.mid_step
                    else:
                        vad_labels[int(self.rlt[i])] = vad_labels[int(self.rlt[i])] + (end_time - each_t)

                    if each_t - self.mid_step < start_time and each_t - self.mid_step >= 0:
                        vad_labels[int(self.rlt[i-1])] = vad_labels[int(self.rlt[i-1])] + (each_t - start_time)
                elif each_t >= end_time:
                    break
                    
            vad_found_label = vad_labels.index(np.max(vad_labels))

            new_vad.append({
                'start_time': start_time,
                'end_time': end_time,
                'speaker': vad_found_label
            })
            
        with open(output_filepath, 'w') as f:
            json.dump(new_vad, f)

        self.new_vad = new_vad
            
        return new_vad
                  
    def evaluate_diarization_accuracy(self, ground_truth_filepath):
        """
        This function calculate accuracy of prediction comparing to ground truth
        
        Parameters
        ----------
        ground_truth_filepath : str
            filepath to the ground trush file.

        Returns
        -------
        accuracy : float
            accuracy between 0 to 1
        
        """
        gt = {}
        with open(ground_truth_filepath, 'rb') as f:
            gt['raw_data'] = json.load(f)
            
        match_time = 0
        total_time = 0
        for i in range(len(gt['raw_data'])):
            if self.new_vad[i]['speaker'] == gt['raw_data'][i][2]:
                match_time = match_time + self.new_vad[i]['end_time'] - self.new_vad[i]['start_time']

            total_time = total_time + self.new_vad[i]['end_time'] - self.new_vad[i]['start_time']
        
        accuracy = match_time/total_time
        print("Accuracy: {0:.1f}%".format(accuracy*100))
        
        return match_time/total_time
        
    def evaluate_diarization_purity(self, ground_truth_filepath):
        """
        This function calculate purity of the clustering
        
        Parameters
        ----------
        ground_truth_filepath : str
            filepath to the ground trush file.

        Returns
        -------
        purity_cluster_m : float
            avergy cluster purity between 0 to 1
        
        purity_speaker_m : float
            avergy speaker purity between 0 to 1
        """
        gt = {}
        with open(ground_truth_filepath, 'rb') as f:
            gt['raw_data'] = json.load(f)

        gt['start_time'] = []
        gt['end_time'] = []
        gt['labels'] = []
        for each_gt in gt['raw_data']:
            gt['start_time'].append(each_gt[0])
            gt['end_time'].append(each_gt[1])
            gt['labels'].append(each_gt[2])

        self.gt, self.gt_speakers = aS.segments_to_labels(gt['start_time'],
                                                         gt['end_time'],
                                                         gt['labels'],
                                                         self.mid_step)
        self.gt_timestamp = np.array(range(len(self.gt))) * self.mid_step
        self.purity_cluster_m, self.purity_speaker_m = aS.evaluate_speaker_diarization(self.rlt, self.gt)
                  
        print("Speaker purity: {0:.1f}% - Cluster purity: {1:.1f}%".format(100 * self.purity_speaker_m, 100 * self.purity_cluster_m))    
        
        return self.purity_cluster_m, self.purity_speaker_m
                  
    def plot_diarization(self):
        """
        This function plot the predicted speakers along the time and/or the ground truth one if it is avaialbe.
        """
        if self.rlt is None:
            print('Please run speaker_diarization first.')
            return
        
        cluster_names = ["speaker{0:d}".format(c) for c in range(self.num_speakers)]

        fig = plt.figure(figsize=[10, 5])

        ax1 = fig.add_subplot(211)
        ax1.set_title('Prediction')
        ax1.plot(self.rlt_timestamp, self.rlt, 'b')
        ax1.set_yticks(np.array(range(len(cluster_names))))
        ax1.axis((0, self.duration, -1, len(cluster_names)))
        ax1.set_yticklabels(cluster_names)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Speakers')
        
        if self.gt is not None:
            speaker_names = ["speaker{0:d}".format(c) for c in range(len(self.gt_speakers))]
                  
            ax2 = fig.add_subplot(212)
            ax2.set_title('Ground Truth')
            ax2.plot(self.gt_timestamp, self.gt, 'r')
            ax2.set_yticks(np.array(range(len(speaker_names))))
            ax2.axis((0, self.duration, -1, len(speaker_names)))
            ax2.set_yticklabels(speaker_names)
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Speakers')

        fig.tight_layout()
                
                  
    
                  