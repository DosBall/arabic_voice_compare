#import pyaudio
from correlation import correlate
import os
import wave
import sys
from audio_similarity1 import AudioSimilarity
#sys.path.insert(1, 'audio-similarity/audio_similarity')
#from audio_similarity import AudioSimilarity



#correlate("audios1/my_do.mp3", "audios1/mubarak_do.mp3")
#correlate("audios1/music1.mp3", "audios1/music2.mp3")
#correlate("audios1/music1.wav", "audios1/music2.wav")
#correlate("audios1/rec1_10_1.wav", "audios1/rec1_10_2.wav")
print('Match with correlation of %.2f%% at offset %i' % correlate("audios1/rec2_60_1.wav", "audios1/rec2_60_2.wav"))
#correlate("audios1/rec3_25_1.wav", "audios1/rec3_25_2.wav")

#correlate("audios1/music1.mp3", "audios1/rec2_60_2.wav")
#correlate("audios1/rec2_60_2.wav", "audios1/music1.mp3")


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


#myaudio_comapre("audios1/rec3_25_1.wav", "audios1/rec3_25_2.wav")
#myaudio_comapre("audios1/rec2_60_1.wav", "audios1/rec2_60_2.wav")
#myaudio_comapre("audios1", "audios1/rec2_60_2.wav")


#python ./AudioCompare/main.py -f audios1/rec2_60_1.wav -f audios1/rec2_60_2.wav


#python ./comp123/eval.py pis --fake audios1/rec2_60_2.wav -d audios1/output1
#python eval.py <pkid, ikid or fad> --real <path_to_real_data> --fake <path_to_fake_data> -d <output_path>
#python ./comp123/eval.py pkid --real audios1/rec2_60_1.wav --fake audios1/rec2_60_2.wav -d audios1/output1




