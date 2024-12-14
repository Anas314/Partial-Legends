import numpy as np
import librosa
import pywt
import soundfile as sf
import matplotlib.pyplot as plt

def wavelet_denoise(audio_signal, wavelet, level=0, threshold='soft'):
    """
    Perform wavelet-based denoising on an audio signal.

    :param audio_signal: The audio signal (1D numpy array)
    :param wavelet: The type of wavelet to use (default: 'db8')
    :param level: The level of decomposition for the wavelet transform (default: 4)
    :param threshold: Thresholding method ('soft' or 'hard')
    :return: The denoised audio signal
    """
    # Decompose the signal into wavelet coefficients
    coeffs = pywt.wavedec(audio_signal, wavelet, level=level)

    # Apply thresholding to the detail coefficients
    thresholded_coeffs = []

    for i, coeff in enumerate(coeffs):
        if i == 0:
            # Approximation coefficients (low-frequency part), keep unchanged
            thresholded_coeffs.append(coeff)
        else:
            # Detail coefficients (high-frequency part), apply thresholding
            if threshold == 'soft':
                thresholded_coeffs.append(pywt.threshold(coeff, np.median(np.abs(coeff)) / 0.6745, mode='soft'))
            elif threshold == 'hard':
                thresholded_coeffs.append(pywt.threshold(coeff, np.median(np.abs(coeff)) / 0.6745, mode='hard'))

    # Reconstruct the signal from the thresholded coefficients
    denoised_signal = pywt.waverec(thresholded_coeffs, wavelet)

    # Ensure the denoised signal has the same length as the original signal
    denoised_signal = denoised_signal[:len(audio_signal)]

    return denoised_signal

def plot_audio_signals_overlay(noisy_signal, denoised_signal, sr):
    """
    Plot the noisy and denoised audio signals overlaid on the same graph.

    :param noisy_signal: The noisy audio signal (1D numpy array)
    :param denoised_signal: The denoised audio signal (1D numpy array)
    :param sr: The sampling rate of the audio signal
    """
    # Create a time axis based on the sampling rate
    t = np.linspace(0, len(noisy_signal) / sr, len(noisy_signal))

    # Plot both signals on the same graph
    plt.figure(figsize=(10, 6))

    # Plot noisy signal
    plt.plot(t, noisy_signal, color='r', label='Noisy Audio', alpha=0.6)

    # Plot denoised signal
    plt.plot(t, denoised_signal, color='g', label='Denoised Audio', alpha=0.8)

    # Adding titles and labels
    plt.title('Noisy vs. Denoised Audio')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')

    # Show the plot
    plt.tight_layout()
    plt.show()

def main(input_audio_file, output_audio_file):
    # Load the audio file
    audio_signal, sr = librosa.load(input_audio_file, sr=None)

    # Denoise the audio using wavelet transform (e.g., using sym8)
    denoised_signal = wavelet_denoise(audio_signal, wavelet='sym8')

    # Save the denoised audio to a new file
    sf.write(output_audio_file, denoised_signal, sr)

    print(f"Denoised audio saved to {output_audio_file}")

    # Plot the noisy and denoised signals overlaid on the same graph
    plot_audio_signals_overlay(audio_signal, denoised_signal, sr)

if _name_ == "_main_":
    input_audio_file = '35_Mono.wav'  # Path to input noisy audio file
    output_audio_file = 'out_35_MonoL15.wav'  # Path to save the denoised audio
    main(input_audio_file, output_audio_file)
