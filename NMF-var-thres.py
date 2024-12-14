import librosa
import numpy as np
from sklearn.decomposition import NMF
import soundfile as sf

def apply_mel_nmf(y, sr, n_components=20, consistency_threshold_W=1.5, consistency_threshold_H=1, n_mels=128, n_fft=2048, hop_length=512):
    # Compute the magnitude spectrogram
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(D), np.angle(D)
    
    # Convert the magnitude spectrogram to Mel scale
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_magnitude = np.dot(mel_basis, magnitude)
    
    # Apply NMF on the Mel magnitude spectrogram
    model = NMF(n_components=n_components, init='random', max_iter=2000, random_state=0, solver="mu")
    W_mel = model.fit_transform(mel_magnitude.T)
    H_mel = model.components_
    
    # Identify consistent components by filtering based on parent component mean
    W_filtered = np.zeros_like(W_mel)
    H_filtered = np.zeros_like(H_mel)
    
    for i in range(W_mel.shape[1]):
        local_mean_W = np.mean(W_mel[:, i])
        local_mean_H = np.mean(H_mel[i, :])
        consistency_mask_W = W_mel[:, i] > (local_mean_W * consistency_threshold_W)
        consistency_mask_H = H_mel[i, :] > (local_mean_H * consistency_threshold_H)
        W_filtered[:, i] = np.where(consistency_mask_W, W_mel[:, i], 0)
        H_filtered[i, :] = np.where(consistency_mask_H, H_mel[i, :], 0)
    
    # Convert the denoised Mel spectrogram back to the linear scale
    denoised_mel_magnitude = np.dot(W_filtered, H_filtered).T
    inv_mel_basis = np.linalg.pinv(mel_basis)
    denoised_magnitude = np.dot(inv_mel_basis, denoised_mel_magnitude)
    
    # Combine magnitude with the original phase
    denoised_D = denoised_magnitude * np.exp(1j * phase)
    
    # Inverse STFT to convert back to time domain
    denoised_y = librosa.istft(denoised_D, hop_length=hop_length)
    
    return denoised_y

def normalize_audio(audio):
    max_amplitude = np.max(np.abs(audio))  # Find max amplitude
    if max_amplitude == 0:
        return audio
    return audio / max_amplitude

# Load the audio file
audio_path = "ntts1-w0.2.wav"                    #       <-----  path for audio wav here
y, sr = librosa.load(audio_path, sr=None)

# Apply Mel scale NMF denoising
denoised_y = apply_mel_nmf(y, sr, n_components=20, consistency_threshold_W=2.0)
denoised_y = normalize_audio(denoised_y)

# Save the denoised audio
output_path = audio_path[:-4] + "-NMF-var-denoised.wav"
sf.write(output_path, denoised_y, sr)

# Play the denoised audio
import sounddevice as sd
sd.play(denoised_y, sr)
sd.wait()  # Wait until the audio finishes playing
