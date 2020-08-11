from speakerDiarization import speakerDiarization as sD

# parameters
audio_filepath = './data/test1.wav'
vad_filepath = './data/test1_vad.json'
ground_truth_filepath = './data/test1_gt.json'
num_speakers = 2
output_vad_filepath = './data/new_vad.json'


# create the instance of the class
sd = sD.speakerDiarization(audio_filepath=audio_filepath,
                           vad_filepath=vad_filepath,
                           num_speakers=2)

# calculate the new vad output file
new_vad = sd.speaker_diarization(output_vad_filepath)

# Accuracy comparing to the ground truth file
accuracy = sd.evaluate_diarization_accuracy(ground_truth_filepath)

# Average cluster and speaker purity
average_cluster_purity, average_speaker_purity = sd.evaluate_diarization_purity(ground_truth_filepath)