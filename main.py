import numpy as np
from matplotlib import pyplot as plt
from scipy.io.wavfile import read, write
import numpy.fft as FFT


# Fonction pour créer une fenêtre de Hamming
def fenetrageHamming(size):
    hamming = np.zeros(size)
    for i in range(size):
        hamming[i] = 0.54 - 0.46 * np.cos((2 * np.pi * i) / size)
    return hamming

def transformeeFourier():
    fft_size = 1024
    # Calculer la FFT
    spectre_fenetre = np.fft.fft(signal_fenetre_fenetre, fft_size)

    # Calculer la transformée inverse
    return np.real(np.fft.ifft(spectre_fenetre, fft_size))


# Ouvrir un fichier wav
filename = "test_seg.wav"
fs, signal = read(filename)

# Informations sur le signal
n_samples = len(signal)
n_milliseconds = (n_samples / fs) * 1000
print("Fréquence d'échantillonnage :", fs)
print("Taille du fichier en échantillons :", n_samples)
print("Taille du fichier en millisecondes :", n_milliseconds)

# Paramètres de la boucle OLA
fen_size = int(fs * 0.032)  # Taille de la fenêtre de 32ms
hop_size = int(fs * 0.008)  # Taille de pas de 8ms
n_fen = int((n_samples - fen_size) / hop_size) + 1  # Nombre de fenêtres

# Créer la fenêtre de Hamming
hamming = fenetrageHamming(fen_size)


# Boucle pour appliquer la fenêtre de Hamming à chaque morceau de signal
for i in range(n_fen):
    # Extraire le morceau de signal
    fen_debut = i * hop_size
    fen_fin = fen_debut + fen_size
    signal_fenetre = signal[fen_debut:fen_fin]

    # Appliquer la fenêtre de Hamming
    signal_fenetre_fenetre = np.multiply(signal_fenetre, hamming)

    # Appliquer la FFT et la transformée inverse
    signal_fenetre_modif = transformeeFourier()



    # Ajouter le morceau de signal modifié au signal final
    if i == 0:
        signal_modif = np.zeros(n_samples)
        somme_hamming = np.ones(n_samples) * 1e-6
    signal_modif[fen_debut:fen_fin] += signal_fenetre_modif[:fen_size] / np.sum(hamming)
    somme_hamming[fen_debut:fen_fin] += hamming

# Normaliser le signal modifié
signal_modif /= somme_hamming

# Écrire le fichier wav
write("resultat.wav", fs, np.int16(signal_modif))