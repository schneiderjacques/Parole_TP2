import numpy as np
from matplotlib import pyplot as plt
from scipy.io.wavfile import read, write
import numpy.fft as FFT


def fenetrageHamming(size):
    return np.hamming(size)


def amplitude_spectrum(signal, fft_size=1024):
    return np.abs(np.fft.fft(signal, fft_size))


def phase_spectrum(signal, fft_size=1024):
    return np.angle(np.fft.fft(signal, fft_size))


def minima_locaux(spectre, intervalle_frequences):
    minima = []
    for i in range(1, len(spectre) - 1):
        if spectre[i - 1] > spectre[i] < spectre[i + 1]:
            minima.append(spectre[i])
        else:
            minima.append(np.nan)

    if len(minima) > intervalle_frequences:
        minima = minima[:intervalle_frequences]

    minima_padded = np.pad(minima, (0, intervalle_frequences - len(minima)), 'constant',
                           constant_values=(np.nan, np.nan))
    return minima_padded



def interpoler_bruit(bruit_estime, fft_size):
    n_minima_locaux = len(bruit_estime)
    bruit_interpole = np.interp(np.arange(0, fft_size), np.linspace(0, fft_size, n_minima_locaux), bruit_estime)
    return bruit_interpole


def soustraction_spectrale(spectre_amplitude, bruit_interpole, alpha, beta, coef):
    return np.maximum(spectre_amplitude - alpha * bruit_interpole, coef * spectre_amplitude ** beta)


def reconstruction_signal(spectre_amplitude, spectre_phase, hop_size):
    n_fen, fft_size = spectre_amplitude.shape
    n_samples = (n_fen - 1) * hop_size + fft_size

    signal_reconstruit = np.zeros(n_samples)

    for i in range(n_fen):
        fen_debut = i * hop_size
        fen_fin = fen_debut + fft_size

        # Remplir la fin de la fenêtre avec des zéros pour avoir la même longueur
        signal_fenetre = np.zeros(n_samples)
        signal_fenetre[fen_debut:fen_fin] = np.real(np.fft.ifft(spectre_amplitude[i] * np.exp(1j * spectre_phase[i])))

        signal_reconstruit += signal_fenetre

    return signal_reconstruit



def main():
    # Paramètres du traitement
    fft_size = 1024
    intervalle_frequences = 102
    alpha = 2
    beta = 1
    coef = 0

    # Lecture du fichier audio
    filename = "test_seg_bruit_10dB.wav"
    fs, signal = read(filename)

    # Taille de la fenêtre et taille de pas
    fen_size = int(fs * 0.032)  # Taille de la fenêtre de 32ms
    hop_size = int(fs * 0.008)  # Taille de pas de 8ms

    # Nombre de fenêtres
    n_samples = len(signal)
    n_fen = int((n_samples - fen_size) / hop_size) + 1

    # Fenêtre de Hamming
    hamming = fenetrageHamming(fen_size)

    # Tableaux pour stocker les spectres d'amplitude et de phase
    spectre_amplitude_avant = np.zeros((n_fen, fft_size))
    phase_avant = np.zeros((n_fen, fft_size))

    # Tableau pour stocker les minima locaux des spectres d'amplitude
    minima_locaux_spectres = np.zeros((n_fen, fft_size // intervalle_frequences, 102))

    # Tableau pour stocker les spectres d'amplitude modifiés
    spectre_amplitude_modifie = np.zeros((n_fen, fft_size))

    # Boucle sur les fenêtres
    for i in range(n_fen):
        # Début et fin de la fenêtre
        fen_debut = i * hop_size
        fen_fin = fen_debut + fen_size

        # Fenêtre temporelle et fenêtre de Hamming
        signal_fenetre = signal[fen_debut:fen_fin]
        signal_fenetre_fenetre = np.multiply(signal_fenetre, hamming)

        # Calcul des spectres d'amplitude et de phase
        spectre_amplitude = amplitude_spectrum(signal_fenetre_fenetre, fft_size)
        spectre_amplitude_avant[i] = spectre_amplitude
        phase_avant[i, :] = phase_spectrum(signal_fenetre_fenetre, fft_size)

        # Estimation des minima locaux des spectres d'amplitude
        minima_locaux_spectres[i, :] = minima_locaux(spectre_amplitude, intervalle_frequences)

    # Interpolation du bruit estimé pour avoir un spectre complet
    minima_locaux_global = np.nanmean(minima_locaux_spectres, axis=0)
    bruit_interpole = interpoler_bruit(minima_locaux_global.flatten(), fft_size)

    # Soustraction spectrale
    for i in range(n_fen):
        # Soustraction spectrale
        spectre_amplitude_modifie[i] = soustraction_spectrale(spectre_amplitude_avant[i], bruit_interpole, alpha, beta, coef)

    # Reconstruction du signal modifié
    signal_modifie = reconstruction_signal(spectre_amplitude_modifie, phase_avant, hop_size)

    # Écriture du fichier wav reconstruit
    write("resultat_reconstruit.wav", fs, np.int16(signal_modifie))

#Launch main function
if __name__ == "__main__":
    main()