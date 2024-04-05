#!/usr/bin/env python3
#chromaprint
#https://github.com/acoustid/chromaprint
#https://oxygene.sk/2011/01/how-does-chromaprint-work/
import argparse
import json
import subprocess
from functools import lru_cache, partial
from operator import sub, itemgetter
from typing import Dict, List, Tuple

from Bio.pairwise2 import align
#from Bio.Align import PairwiseAligner
import os
from pydub import AudioSegment
import time
import wave
import struct
import librosa
import soundfile as sf
from correlation import correlate
from pydub.silence import split_on_silence


def get_grey_code(n: int) -> List[str]:
    #Get n-bit grey code
    if n <= 0:
        raise ValueError("must be positive integer")

    if n == 1:
        return ['0', '1']

    l = get_grey_code(n - 1)
    l1 = ['0' + e for e in l]
    l2 = ['1' + e for e in l[::-1]]

    return l1 + l2


GREY_CODE: Dict[int, int] = {int(c, 2): i for i, c in enumerate(get_grey_code(2))}


def get_fingerprint(p):
    #Use chromaprint to calculate fingerprint from file path
    out = subprocess.run(['fpcalc', '-json', '-raw', p], capture_output=True, check=True)
    return json.loads(out.stdout)


def get_chunk(a: int, i: int):
    return (a >> (2 * i)) & 0b11


@lru_cache(maxsize=2048)
def get_chunks(a: int) -> Tuple[int]:
    #Splitting integer into 16 2-bit chunks
    return itemgetter(*map(partial(get_chunk, a), range(16)))(GREY_CODE)


def match_score(a: int, b: int) -> int:
    #Get absolute difference of grey code between chunks. <8 is match and >40 is mismatch.
    #No penalty otherwise. Theoretical minimum in grey difference is 0, maximum is 48.
    score = sum(map(abs, map(sub, get_chunks(a), get_chunks(b))))
    if score < 16:
        return 1
    if score > 32:
        return -1
    return 0


def similarity_score(fp1: List[int], fp2: List[int]) -> float:
    #Get similarity score between fingerprints
    #Use global alignment to align fingerprints, with custom match score calculator.
    aln = align.globalcs(fp1, fp2, match_score, -2, 0, gap_char=[627964279])[0]
    #aln = PairwiseAligner.align.globalcs(fp1, fp2, match_score, -2, 0, gap_char=[627964279])[0]
    return aln[2] / aln[4]


def my_silence_del(file1):  # удаление тишины из аудио
    """
    if file1[-3:].lower() == "wav":
        source = wave.open(file1, mode="rb")
        dest = wave.open(file1[:-4] + "_0.wav", mode="wb")
        dest.setparams(source.getparams())
        frames_count = source.getnframes()
        data = struct.unpack("<" + str(frames_count) + "h",
                             source.readframes(frames_count))
        newdata = list(filter(lambda x: abs(x) > 5, data))
        newframes = struct.pack("<" + str(len(newdata)) + "h", *newdata)
        dest.writeframes(newframes)
        source.close()
        dest.close()
        return file1[:-4] + "_0.wav"
        #sf.write(file1[:-4] + "_0.mp3", new_audio, samplerate=int(sample_rate))
    elif file1[-3:].lower() == "mp3":
    """
    source = open(file1, mode="rb")
    audio = AudioSegment.from_file(source)
    silence_threshold = -40  # in decibels
    segments = split_on_silence(audio, silence_thresh=silence_threshold)
    non_silent_audio = sum(segments)
    file_out = file1[:-4] + "_0" + file1[-4:]
    non_silent_audio.export(file_out, format=file1[-3:])
    return file_out


def dlina_audio(file1, rate1):  # растягиваем длину меньшего аудио
    audio, sample_rate = librosa.load(file1)
    #print("sample_rate:", sample_rate)
    new_audio = librosa.effects.time_stretch(y=audio, rate=rate1)
    file_out = file1[:-4] + "_1" + file1[-4:]
    sf.write(file_out, new_audio, samplerate=int(sample_rate))
    return file_out


def cloneaudio(file1, n1):  # копируем аудио n1 раз и "склеиваем" копии
    if file1[-3:].lower() == "mp3":
        audio1 = AudioSegment.from_mp3(file1)
        for i in range(n1):
            audio1 += AudioSegment.from_mp3(file1)
        new1 = file1[:-4] + "_2.mp3"
        audio1.export(new1, format="mp3")
        #print("Новый файл: " + new1)
        return new1
    elif file1[-3:].lower() == "wav":
        audio1 = AudioSegment.from_wav(file1)
        for i in range(n1):
            audio1 += AudioSegment.from_wav(file1)
        new1 = file1[:-4] + "_2.wav"
        audio1.export(new1, format="wav")
        #print("Новый файл: " + new1)
        return new1
    else:
        return file1


def compare1(file1, file2):
    start = time.time()
    if file1[-3:].lower() not in ["mp3", "wav"]:  # проверяем формат аудио
        print("Неподходящий формат аудио. Используйте mp3 либо wav")
    file1, file2 = my_silence_del(file1), my_silence_del(file2)  # удаляем тишину из аудио
    size1, size2 = os.path.getsize(file1), os.path.getsize(file2)
    #'''
    if size1 > size2:  # растягиваем меньшее аудио до большего
        file2 = dlina_audio(file2, size2 / size1)
        file1 = dlina_audio(file1, 1.0001)
    elif size2 > size1:
        file1 = dlina_audio(file1, size1 / size2)
        file2 = dlina_audio(file2, 1.0001)
    #'''
    if size1 < 1000000 or size2 < 1000000:  # если аудио слишком маленькие, копируем их и "склеиваем" копии
        n1 = 1000000 // min(size1, size2)
        file1 = cloneaudio(file1, n1)
        file2 = cloneaudio(file2, n1)
    data1, data2 = get_fingerprint(file1), get_fingerprint(file2)
    score = similarity_score(data1["fingerprint"], data2["fingerprint"])  # считаем similarity
    print(f"Similarity score: {score}")
    print('Match with correlation of %.2f%% at offset %i' % correlate(file1, file2))  # доп. простой способ подсчета % с помощью fpcalc
    print("Время: ", str(time.time() - start), "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs=2, help="files to compair")
    args = parser.parse_args()
    compare1(args.file[0], args.file[1])



