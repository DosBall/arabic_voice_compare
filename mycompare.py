#import pyaudio
from correlation import correlate
import os
import wave
import sys
from audio_similarity1 import AudioSimilarity

def myaudio_comapre(original_path, compare_path):
    if original_path[-4:] in [".mp3", ".wav", ".ogg"]:
        with wave.open(original_path, "rb") as wave_file1:
            frame_rate1 = wave_file1.getframerate()
            sample_rate = frame_rate1
            print("rate1 =", frame_rate1)
    if compare_path[-4:] in [".mp3", ".wav", ".ogg"]:
        with wave.open(compare_path, "rb") as wave_file2:
            frame_rate2 = wave_file2.getframerate()
            sample_rate = frame_rate2
            print("rate2 =", frame_rate2)
    weights = {
        'zcr_similarity': 0.2,
        'rhythm_similarity': 0.2,
        'chroma_similarity': 0.2,
        'spectral_contrast_similarity': 0.1,
        'perceptual_similarity': 0.2
    }

    audio_similarity = AudioSimilarity(original_path, compare_path, sample_rate, weights, verbose=True)#, sample_size=20
    print(f"Audio Similarity: {audio_similarity}")
    zcr_similarity = audio_similarity.zcr_similarity()
    print(f"ZCR Audio Similarity: {zcr_similarity}")
    similarity_score = audio_similarity.stent_weighted_audio_similarity(metrics='all') # You can select all metrics or just the 'swass' metric
    print(f"Stent Weighted Audio Similarity Score: {similarity_score}")


#python ./AudioCompare/main.py -f audios1/rec2_60_1.wav -f audios1/rec2_60_2.wav


