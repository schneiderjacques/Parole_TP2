import numpy as np
from matplotlib import pyplot as plt
from scipy.io.wavfile import read, write


# Fonction pour créer une fenêtre de Hamming
def fenetrageHamming(size):
    hamming = np.zeros(size)
    for i in range(size):
        hamming[i] = 0.54 - 0.46 * np.cos((2 * np.pi * i) / size)
    return hamming


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



# Écrire le fichier wav
#write("resultat.wav", fs, np.int16(signal_modif))
