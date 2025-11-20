# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import sys


# ============================================================
# Backend de audio (compartido con el transmisor analógico)
# ------------------------------------------------------------
# Usa sounddevice si está disponible; si falla, intenta PyAudio.
# Opcionalmente duplica a estéreo y aplica ganancia en dB.
# ============================================================

_AUDIO_BACKEND_SD = None
_AUDIO_BACKEND_PYA = None

def _backend_sd_play(y, fs):
    global _AUDIO_BACKEND_SD
    if _AUDIO_BACKEND_SD is None:
        import sounddevice as sd
        _AUDIO_BACKEND_SD = sd
    sd = _AUDIO_BACKEND_SD
    sd.play(y, fs, blocking=True)

def _backend_pyaudio_play(y, fs):
    global _AUDIO_BACKEND_PYA
    if _AUDIO_BACKEND_PYA is None:
        import pyaudio
        _AUDIO_BACKEND_PYA = pyaudio
    pyaudio = _AUDIO_BACKEND_PYA
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=pyaudio.paFloat32,
                        channels=y.shape[1] if y.ndim == 2 else 1,
                        rate=fs,
                        output=True)
        stream.write(y.astype(np.float32).tobytes())
        stream.stop_stream()
        stream.close()
    finally:
        p.terminate()

def play_audio(x, fs, gain_db=0.0, to_stereo=True, device=None, block=True):
    """Reproduce un buffer de audio float con backend SD/PyAudio.

    Mantiene la firma previa (to_stereo, device, block) para no
    cambiar las llamadas existentes, pero internamente usa el
    mismo backend robusto del transmisor analógico.
    """
    y = np.asarray(x, dtype=np.float32)
    if y.ndim == 1 and to_stereo:
        y = np.column_stack([y, y])
    g = 10.0 ** (gain_db / 20.0)
    y = np.clip(y * g, -0.95, 0.95).astype(np.float32)

    # Intentar primero sounddevice, luego PyAudio.
    try:
        _backend_sd_play(y, fs)
        return
    except Exception as e_sd:
        try:
            _backend_pyaudio_play(y, fs)
            return
        except Exception as e_pa:
            print(f"[Audio] No se pudo reproducir: {e_sd} | {e_pa}")
import os
import wave
# Nota: interfaz gráfica removida. La selección de archivo ahora se realiza por ruta manual.

# ---------- Parámetros de los beeps ----------
BEEP_FREQ = 1000          # Frecuencia del beep (Hz)
BEEP_MS = 200             # Duración de cada beep (ms)
BEEP_GAP_S = 1.0          # Separación entre beeps (s)
BEEP_LEVEL = 0.6          # Nivel del beep (0–1)

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
    
    beep = make_beep(fs, BEEP_FREQ, 1000, BEEP_LEVEL)                                # Genera beep
    gap = make_silence(fs, BEEP_GAP_S)                                              # Genera silencio de 1 s
    seq = np.concatenate([gap, beep, gap])                                          # Concatena beep-gap-beep-gap-beep
    play_audio(seq, fs, gain_db=0.0, to_stereo=True, device=None, block=True)  


######Inicio de ecxtraccion de bits desde archivo######

# Función para extraer bits de un archivo
def extraer_bits_desde_archivo(file_path):
    """
    Lee un archivo y extrae todos sus bits.
    
    Parámetros:
    - file_path: ruta del archivo a leer
    

    Devuelve:
    - bits: array de bits (0 y 1)
    - file_size: tamaño del archivo en bytes
    """
    try:
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        file_size = len(file_data)
        print(f"[Archivo] Tamaño: {file_size} bytes")
        
        # Convertir cada byte a 8 bits
        bits = np.unpackbits(np.frombuffer(file_data, dtype=np.uint8))
        
        print(f"[Bits] Total de bits extraídos: {len(bits)}")
        return bits, file_size
    
    except Exception as e:
        print(f"[Error] No se pudo leer el archivo: {e}")
        return None, None

# Función para generar una cadena de bits aleatoria en NRZ unipolar (para pruebas)
def generar_cadena_bits_nrz_unipolar(bit_length=8192):
    # Generar una cadena de bits aleatorios (0 o 1)
    bits = np.random.randint(0, 2, size=bit_length)

    # Convertir la cadena de bits en NRZ unipolar: 0 -> 0, 1 -> 1
    nrz_signal = np.array([bit if bit == 1 else 0 for bit in bits])

    return bits, nrz_signal

# Función para graficar la señal NRZ unipolar
def graficar_nrz_signal(nrz_signal):
    plt.figure(figsize=(10, 4))
    plt.step(np.arange(len(nrz_signal)), nrz_signal, where='post', color='blue', linewidth=2)
    plt.ylim(-0.5, 1.5)
    plt.title("Señal NRZ unipolar")
    plt.xlabel("Índice de bit")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.show()

# Función para modular en BPSK a partir de una señal NRZ unipolar
def modula_bpsk(nrz_unipolar, fc=10_000, fs=200_000, sps=40):
    """
    Modula una secuencia NRZ unipolar (0/1) en BPSK con portadora de amplitud 1.

    Parámetros:
    - nrz_unipolar: array de 0/1 por bit
    - fc: frecuencia de la portadora (< 25 kHz)
    - fs: frecuencia de muestreo (Hz)
    - sps: muestras por símbolo (bits)

    Devuelve:
    - t: vector de tiempo
    - s: señal BPSK modulada en tiempo
    """
    if fc >= 25_000:
        raise ValueError("fc debe ser < 25 kHz")
    # Mapear 0->-1 y 1->+1 (NRZ bipolar)
    symbols = 2 * np.array(nrz_unipolar, dtype=float) - 1.0
    # Expandir cada símbolo a sps muestras
    baseband = np.repeat(symbols, sps)
    # Tiempo
    t = np.arange(len(baseband)) / fs
    # Portadora de amplitud 1
    carrier = np.cos(2 * np.pi * fc * t)
    # Modulación BPSK
    s = baseband * carrier
    return t, s

# Función para graficar la señal BPSK en el tiempo
def graficar_bpsk_tiempo(t, s):
    plt.figure(figsize=(12, 4))
    plt.plot(t, s, color='crimson', linewidth=1.0)
    plt.title("Señal BPSK en el tiempo")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Función para graficar el espectro (magnitud) de la señal BPSK
def graficar_espectro_bpsk(s, fs, fc=None, xlim_khz=None):
    N = len(s)
    if N == 0:
        raise ValueError("La señal BPSK está vacía")
    # Ventana para reducir leakage
    w = np.hanning(N)
    s_win = s * w
    S = np.fft.rfft(s_win)
    f = np.fft.rfftfreq(N, d=1.0 / fs)
    # Magnitud normalizada en dB (0 dB = pico)
    mag = np.abs(S)
    mag_db = 20 * np.log10(mag / (mag.max() + 1e-12) + 1e-12)

    plt.figure(figsize=(12, 4))
    plt.plot(f / 1e3, mag_db, color='teal', linewidth=1.0, label='|S(f)|')
    if fc is not None:
        plt.axvline(fc / 1e3, color='orange', linestyle='--', linewidth=1.0, label='fc')
    if xlim_khz is not None:
        plt.xlim(0, xlim_khz)
    plt.title("Espectro de la señal BPSK (normalizado)")
    plt.xlabel("Frecuencia [kHz]")
    plt.ylabel("Magnitud [dB]")
    plt.grid(True)
    if fc is not None:
        plt.legend()
    plt.tight_layout()
    plt.show()

# Utilidad: remuestreo lineal simple a fs_out para compatibilidad con la tarjeta de sonido
def resample_linear(x, fs_in, fs_out):
    if fs_in == fs_out:
        return x
    n_in = len(x)
    dur = n_in / fs_in
    n_out = int(round(dur * fs_out))
    if n_out <= 1:
        return np.array([], dtype=x.dtype)
    t_in = np.linspace(0.0, dur, n_in, endpoint=False)
    t_out = np.linspace(0.0, dur, n_out, endpoint=False)
    y = np.interp(t_out, t_in, x)
    return y

# Utilidad: guardar WAV PCM16
def save_wav_pcm16(path, x, fs):
    x_clip = np.clip(x, -1.0, 1.0)
    x_i16 = (x_clip * 32767.0).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(fs)
        wf.writeframes(x_i16.tobytes())

def reproducir_bpsk(s, fs_in, fs_out=48_000, gain_db=-3.0):
    if len(s) == 0:
        print("[TX] Señal vacía; nada que reproducir.")
        return

    # Remuestreo lineal a la frecuencia de muestreo de salida
    y = resample_linear(s, fs_in, fs_out)
    if len(y) == 0:
        print("[TX] Remuestreo falló; longitud de salida cero.")
        return

    # Enviar al backend de audio compartido (misma lógica que el TX analógico)
    try:
        play_audio(y, fs_out, gain_db=gain_db, to_stereo=True, device=None, block=True)
        print("[TX] Reproducción BPSK finalizada.")
    except Exception as e:
        print(f"[TX] Error al reproducir BPSK: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("TRANSMISOR DIGITAL CON MODULACIÓN BPSK")
    print("=" * 60)
    print("\nModo de entrada: ingrese la ruta completa del archivo para transmitir")
    print("(deje vacío y presione Enter para usar bits aleatorios de prueba)")

    if len(sys.argv) > 1:
        file_path = sys.argv[1].strip()
        print(f"\n[TX] Usando archivo desde línea de comandos: {file_path}")
    else:
        file_path = input("\nRuta del archivo (o dejar vacío): ").strip()


    if file_path:
        # Usar archivo indicado por ruta manual
        if not os.path.exists(file_path):
            print(f"[TX][Error] La ruta indicada no existe: {file_path}")
            sys.exit(1)

        bits, file_size = extraer_bits_desde_archivo(file_path)
        if bits is None:
            sys.exit(1)

        print(f"[TX] Archivo: {os.path.basename(file_path)} cargado exitosamente")
        nrz_signal = bits  # Los bits del archivo ya están en formato 0/1
    else:
        # Usar bits aleatorios
        print("\n[TX] Generando cadena de bits aleatoria de prueba...")
        bits, nrz_signal = generar_cadena_bits_nrz_unipolar(bit_length=8192)
    
    # Mostrar los resultados en consola
    print(bits)

    print("\n[TX] Cadena de bits generada (primeros 64 bits):", bits[:64])
    print(f"[TX] Cantidad total de bits: {len(bits)}")

    # Modulación BPSK (fc < 25 kHz, amplitud 1)
    fc = 10_000      # Hz
    fs = 200_000     # Hz
    sps = 40         # muestras por símbolo
    print(f"\n[TX] Parámetros de modulación:")
    print(f"[TX]   Frecuencia portadora: {fc} Hz")
    print(f"[TX]   Frecuencia de muestreo: {fs} Hz")
    print(f"[TX]   Muestras por símbolo: {sps}")
    
    t, s_bpsk = modula_bpsk(nrz_signal, fc=fc, fs=fs, sps=sps)
    
    print(f"[TX] Duración de la señal modulada: {len(s_bpsk)/fs:.3f} segundos")

    # Reproducir la señal modulada por parlantes
    print("\n[TX] Iniciando reproducción...")
    fs_out = 48_000
    play_triple_beep(fs_out)
    reproducir_bpsk(s_bpsk, fs, fs_out=fs_out, gain_db=-6.0)
    play_beep_final(fs_out)
    
       # Graficar la señal NRZ
    graficar_nrz_signal(nrz_signal)

    # Graficar la señal BPSK modulada en el tiempo
    graficar_bpsk_tiempo(t, s_bpsk)

    # Graficar solo el espectro de la señal BPSK
    graficar_espectro_bpsk(s_bpsk, fs, fc=fc, xlim_khz=50)
