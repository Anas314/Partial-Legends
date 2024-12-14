##-------------------------------------- PESQ TEST ------------------------------------##

from pesq import pesq  #needs pesq library
from scipy.io import wavfile
import numpy as np

# Load reference and degraded audio files
fs_ref, ref_audio = wavfile.read("lec8.wav")  # Load reference audiaudio (original with no noise)
fs_deg, deg_audio = wavfile.read("#rr.wav")  # Load degraded audio (denoised audio your alogarithm output)

# Ensure both files have the same sampling rate
if fs_ref != fs_deg:
    raise ValueError("Sampling rates of the reference and degraded files do not match!")

# Use PESQ to compute the score
# 'wb' for wideband (16 kHz), 'nb' for narrowband (8 kHz)
pesq_score = pesq(fs_ref, ref_audio, deg_audio, mode='wb')

print(f"PESQ Score: {pesq_score}")
