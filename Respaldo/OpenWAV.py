# ===============================================================
# TRANSMISOR SSB / ISB – Comentado (mantiene el código intacto)
# ---------------------------------------------------------------
# Grupos de funciones:
#  1) E/S y reproducción de audio  -> read_wav_mono, play_audio
#  2) Modulación SSB/ISB           -> ssb_modulate
#  3) Señales de control (handshake)-> make_beep, make_silence,
#                                     play_triple_beep, play_beep_final
#  4) main()
# ===============================================================================================================================

import os
import numpy as np
import soundfile as sf
import sounddevice as sd
from scipy.signal import hilbert

# ========= RUTA ================================================================================================================
BASE_DIR = r"/home/ciqiu/Proyectos/Proyecto2TallerComu"  # Carpeta del proyecto
FILENAME = "Entrada.wav"  # Nombre del archivo WAV de entrada
# =================================================================================================================================

MAX_AMPL = 0.95  # Límite para evitar clipping al reproducir

# ---------- Parámetros de SSB predeterminados ----------
SSB_FC_HZ = 12_000            # Frecuencia de portadora (Hz) para SSB
SSB_SIDE = "USB"              # Banda a usar: 'USB' o 'LSB'
SSB_FC_ERROR = 0.0            # Error relativo de frecuencia
SSB_PHASE_ERROR_DEG = 0.0     # Error de fase (0–180)
SSB_CARRIER_LEVEL = 0.0       # >0 para SSB-FC (agrega portadora), 0.0 = SSB-SC

# ---------- Parámetros de los beeps ----------
BEEP_FREQ = 1000          # Frecuencia del beep (Hz)
BEEP_MS = 200             # Duración de cada beep (ms)
BEEP_GAP_S = 1.0          # Separación entre beeps (s)
BEEP_LEVEL = 0.6          # Nivel del beep (0–1)

# ===============================================================================================================================
#                   GRUPO 1 — Manejo .WAV
#               Estas funciones leen WAV (mono)
# ===============================================================================================================================

def read_wav_mono(path): # Lee un WAV, lo convierte a mono promediando canales y devuelve (x, fs). También normaliza.
    
    x, fs = sf.read(path, always_2d=True)                                       # Lee WAV 
    x = x.mean(axis=1).astype(np.float64)                                       # Promedia a mono
    peak = np.max(np.abs(x)) if x.size else 1.0                                 # Pico absoluto para normalizar
    if peak > 1.0:                                                              # Si venía saturado lo normaliza
        x = x / peak                                                            # Escala a [-1, 1]
    return x, fs  


def play_audio(x, fs, gain_db=0.0, to_stereo=True, device=None, block=True): # Reproduce la señal x a fs.

    g = 10.0 ** (gain_db / 20.0)                                                # Convierte dB a ganancia lineal
    y = np.clip(x * g, -MAX_AMPL, MAX_AMPL).astype(np.float32)                  # Aplica ganancia y limita
    if to_stereo:                                                               # Si se requiere estéreo, duplica L=R
        y = np.column_stack([y, y])                                             
    try:
        sd.play(y, fs, device=device, blocking=block)                           # Reproduce con bloqueo opcional
    except sd.PortAudioError as e:                                              # Maneja errores de dispositivo
        print(f"[Audio] Error de PortAudio: {e}\nDispositivos disponibles:")    
        print(sd.query_devices())                                               
        raise                                                                   

# ================================================================================================================================
#                   GRUPO 2 — Modulación
# Implementa la modulación de banda lateral única a partir de la
#           señal analítica. Permite USB/LSB y SSB-SC/FC;
#           errores de fase y de frecuencia en la portadora.
# ===============================================================================================================================

def ssb_modulate(x, fs, fc, side='USB', fc_error=0.0, phase_error_deg=0.0, carrier_full_scale=0.0):
    
    xa = hilbert(x)                                                             # Señal analítica: x + j*H{x}
    if side.upper() == 'LSB':                                                   # Si se requiere LSB
        xa = np.conj(xa)                                                        # Conjuga para invertir espectro
    fc_eff = fc * (1.0 + fc_error)                                              # Aplica error relativo de frecuencia
    phi = np.deg2rad(phase_error_deg)                                           # error de fase a radianes
    n = np.arange(len(x))                                                       # Índices de muestra
    osc = np.exp(1j * (2*np.pi*fc_eff*n/fs + phi))                              # Oscilador complejo
    s_sc = np.real(xa * osc)                                                    # Parte real → SSB suprimida de portadora (SC)
    s = s_sc.copy()                                                             # Copia para salida
    if carrier_full_scale > 0.0:                                                # Si se solicita SSB-FC
        s += carrier_full_scale * np.cos(2*np.pi*fc_eff*n/fs + phi)             # Agrega portadora
    s = np.clip(s, -0.95, 0.95)                                                 # Limita para evitar clipping duro
    return s  

# ================================================================================================================================
#                   GRUPO 3 — Handshake
# Genera y reproduce tonos de arranque (3 beeps) y un beep final
#               para coordinar TX/RX por audio.
# ===============================================================================================================================

def make_beep(fs, freq=BEEP_FREQ, ms=BEEP_MS, level=BEEP_LEVEL): # Construye un beep senoidal
    
    N = int(fs * (ms / 1000.0))                                                     # Muestras del beep
    t = np.arange(N) / fs                                                           # Vector de tiempo
    env = np.linspace(0, 1, int(0.02*N))                                            # Envolvente ataque (2%)
    env = np.pad(env, (0, N - len(env)), mode='constant', constant_values=1.0)      # Rellena a 1
    s = level * np.sin(2*np.pi*freq*t) * env                                        # Senoide con ataque suave
    return s.astype(np.float32)                                                     # Devuelve beep en float32

def make_silence(fs, seconds): # Devuelve un vector de ceros (silencio) de duración dada.
    
    N = int(fs * seconds)  
    return np.zeros(N, dtype=np.float32)  

def play_triple_beep(fs): #Reproduce un patrón de 3 beeps (handshake de inicio).
    
    beep = make_beep(fs, BEEP_FREQ, BEEP_MS, BEEP_LEVEL)                            # Genera beeps
    gap = make_silence(fs, BEEP_GAP_S)                                              # Genera silencios
    seq = np.concatenate([beep, gap, beep, gap, beep, gap])                         # Une beep-gap-beep-gap-beep
    play_audio(seq, fs, gain_db=0.0, to_stereo=True, device=None, block=True)       # Reproduce secuencia

def play_beep_final(fs): # Reproduce un beep más largo para indicar fin de transmisión.
    
    beep = make_beep(fs, BEEP_FREQ, 600, BEEP_LEVEL)                                # Genera beep
    gap = make_silence(fs, BEEP_GAP_S)                                              # Genera silencio de 1 s
    seq = np.concatenate([gap, beep, gap])                                          # Concatena beep-gap-beep-gap-beep
    play_audio(seq, fs, gain_db=0.0, to_stereo=True, device=None, block=True)       # Reproduce secuencia


# ===============================================================================================================================
#                   Main
# ===============================================================================================================================

def main():
    wav_path = os.path.join(BASE_DIR, FILENAME)                                     # Construye ruta completa del WAV
    if not wav_path.lower().endswith(".wav"):
        candidate = wav_path + ".wav"
        if os.path.exists(candidate):
            wav_path = candidate

    if not os.path.exists(wav_path):                                                # Verifica existencia del archivo
        raise FileNotFoundError(f"No se encontró el archivo WAV:\n{wav_path}")      # Error si no existe

    x, fs = read_wav_mono(wav_path)                                                 # Lee WAV mono y obtiene fs

    print("=== WAV CARGADO ===")
    print(f"Ruta: {wav_path}")
    print(f"Fs: {fs} Hz")
    print(f"Muestras: {len(x)}")
    print(f"Duración: {len(x)/fs:.2f} s")
    print(f"pico abs: {np.max(np.abs(x)):.3f}")

    print("\n[SSB] Modulando...")                                                   # Mensaje de estado
    s_ssb = ssb_modulate(
        x, fs, SSB_FC_HZ, side=SSB_SIDE,
        fc_error=SSB_FC_ERROR,
        phase_error_deg=SSB_PHASE_ERROR_DEG,
        carrier_full_scale=SSB_CARRIER_LEVEL
    )

    out_ssb_path = os.path.join(BASE_DIR, "ssb_out.wav")                            # Ruta de salida para guardar
    sf.write(out_ssb_path, s_ssb, fs)                                               # Guarda WAV modulado
    print(f"[SSB] Señal modulada guardada en: {out_ssb_path}")

    print("\n[TX] Enviando 3 beeps de arranque...")
    play_triple_beep(fs)                                                            # Reproduce los 3 beeps (handshake)

    print("[TX] Transmitiendo la señal SSB...")
    play_audio(s_ssb, fs, gain_db=0.0, to_stereo=True, device=None, block=True)     # Reproduce SSB por bocina
    print("[TX] Lista transmision")

    print("[TX] Beep final")
    play_beep_final(fs) 
    print("[TX] Final.")

if __name__ == "__main__":
    main()
