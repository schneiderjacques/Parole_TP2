import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from scipy.io.wavfile import read, write

'''
# Fonction pour créer une fenêtre de Hamming
def fenetrage_hamming(size: int) -> ndarray:
    hamming = np.zeros(size)
    for i in range(size):
        hamming[i] = 0.54 - 0.46 * np.cos((2 * np.pi * i) / size)
    return hamming
'''


'''
Fonction pour lire un fichier wav.
Paramètre path chemin du fichier en str
Retourne int
'''
def read_wav(path: str):
    return read(path)


'''
Fonction pour récupérer la fréquence d'échantillonnage.
Paramètre signal
Retourne int 
'''
def get_frequence_echantillonnage(signal: any) -> int:
    return read(signal)[0]


'''
Fonction pour afficher un signal avec matplotlib
Paramètres duration et signal
'''
def show_signal(duration: ndarray, signal):
    time = np.linspace(0, duration, len(signal))
    plt.plot(time, signal)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()


'''
Fonction de la boucle OLA
Paramètre signal_values
'''
def loop_overlap_and_add(signal) -> ndarray:
    N = len(signal)
    duree_segment = 32
    duree_pas = 8

    nb_segments = int(np.ceil((N / duree_pas) / duree_segment))

    segments = np.zeros((nb_segments, duree_segment))

    for i in range(nb_segments):
        debut = i * duree_pas
        fin = debut + duree_segment
        if fin > N:
            fin = N
        segments[i, :fin - debut] = signal[debut:fin]
    return segments

def fenetrage_hamming(size) -> ndarray:
    return np.hamming(size, sym=False)

def fenetrage_signal(size: int, signal_extrait: ndarray, hamming: ndarray):
    signal_fen: ndarray = np.array([])
    for i in range(size):
        signal_fen[i] = signal_extrait[i] * hamming[i]
    return signal_fen


def run(file):
    # charger le fichier wav
    sample_rate, signal = read_wav(file)

    # calculer la durée totale du signal
    duration = len(signal) / sample_rate

    # afficher le signal
    show_signal(duration, signal)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    loop_overlap_and_add(read_wav("../asset/test_seg_bruit_0dB.wav")[1])

    '''
    # Ouvrir un fichier wav
    filename = "../asset/test_seg.wav"
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
    hamming = fenetrage_hamming(fen_size)

    # Écrire le fichier wav
    # write("resultat.wav", fs, np.int16(signal_modif))
    '''