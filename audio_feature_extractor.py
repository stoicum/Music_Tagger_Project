import os
import traceback
import librosa
import numpy as np
import csv

base_path = "archive/Data/genres_original/"
audio_file_folder_path = os.listdir(base_path)
print(audio_file_folder_path)

# create the head for csv file
csv_head = ["filename", "length", "chroma_stft_mean", "chroma_stft_var", "rms_mean", "rms_var",
            "spectral_centroid_mean", "spectral_centroid_var", "spectral_bandwidth_mean", "spectral_bandwidth_var",
            "rolloff_mean", "rolloff_var", "zero_crossing_rate_mean", "zero_crossing_rate_var", "harmony_mean",
            "harmony_var", "tempo"]
for i in range(20):
    csv_head.append(f'mfcc{i + 1}_mean')
    csv_head.append(f'mfcc{i + 1}_var')
csv_head.append("label")

audio_features = []
audio_features.append(csv_head)

for folder_name in audio_file_folder_path:
    # for each genre
    path = base_path + folder_name + "/"
    base_paths = os.listdir(path)

    for audio_file_name in base_paths:
        # load each file in this genre
        print(path + audio_file_name)
        try:
            y, sr = librosa.load(path + audio_file_name, duration=30)
        except Exception as e:
            print("Processing File " + audio_file_name + ". Error occurred: " + str(e))
            traceback.print_stack()
            continue

        features = []
        # filename
        features.append(audio_file_name)

        # length
        features.append(30)  

        # chroma_stft
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        features.append(np.mean(chroma_stft))
        features.append(np.var(chroma_stft))

        # RMS
        rms = librosa.feature.rms(y=y)
        features.append(np.mean(rms))
        features.append(np.var(rms))

        # spectral_centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.append(np.mean(spectral_centroids))
        features.append(np.var(spectral_centroids))

        # spectral_bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features.append(np.mean(spectral_bandwidth))
        features.append(np.var(spectral_bandwidth))

        # rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features.append(np.mean(spectral_rolloff))
        features.append(np.var(spectral_rolloff))

        # zero_crossing_rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        features.append(np.mean(zero_crossing_rate))
        features.append(np.var(zero_crossing_rate))

        # harmony
        harmony = librosa.effects.harmonic(y)
        features.append(np.mean(harmony))
        features.append(np.var(harmony))

        # tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features.append(tempo)

        # MFCC 20
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(20):
            features.append(np.mean(mfccs[i]))
            features.append(np.var(mfccs[i]))

        # target value
        features.append(folder_name)
        
        # save featrues
        audio_features.append(features)

# write in the csv
with open('features_30_sec.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(audio_features)
