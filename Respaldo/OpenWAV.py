import os
import numpy as np
import soundfile as sf
import sounddevice as sd
from scipy.signal import hilbert

# ========= CONFIGURA AQUÍ TU RUTA =========
BASE_DIR = r"C:\Users\acost\OneDrive\Escritorio"  # Carpeta del proyecto
FILENAME = "sheesh.wav"  # Nombre del archivo WAV de entrada
# =========================================

MAX_AMPL = 0.95  # Límite suave para evitar clipping al reproducir

# ---------- Parámetros de SSB predeterminados ----------
SSB_FC_HZ = 12_000            # Frecuencia de portadora (Hz) para SSB
SSB_SIDE = "USB"              # Banda a usar: 'USB' o 'LSB'
SSB_FC_ERROR = 0.0            # Error relativo de frecuencia (p.ej. 0.10 = +10%)
SSB_PHASE_ERROR_DEG = 0.0     # Error de fase en grados (0–180)
SSB_CARRIER_LEVEL = 0.0       # >0 para SSB-FC (agrega portadora), 0.0 = SSB-SC

# ---------- Parámetros de los beeps de arranque ----------
BEEP_FREQ = 1000          # Frecuencia del beep (Hz)
BEEP_MS = 200             # Duración de cada beep (ms)
BEEP_GAP_S = 1.0          # Separación entre beeps (s)
BEEP_LEVEL = 0.6          # Nivel del beep (0–1)

def read_wav_mono(path):
    """Lee WAV, fuerza a mono promediando canales. Devuelve x(float64), fs."""
    x, fs = sf.read(path, always_2d=True)  # Lee WAV y fuerza matriz (N, canales)
    x = x.mean(axis=1).astype(np.float64)  # Promedia a mono y usa float64
    peak = np.max(np.abs(x)) if x.size else 1.0  # Pico absoluto para normalizar
    if peak > 1.0:  # Si venía saturado, normaliza
        x = x / peak  # Escala a [-1, 1]
    return x, fs  # Retorna señal y frecuencia de muestreo

def play_audio(x, fs, gain_db=0.0, to_stereo=True, device=None, block=True):
    """Reproduce el arreglo x a fs. Opcionalmente aplica ganancia y duplica a estéreo."""
    g = 10.0 ** (gain_db / 20.0)  # Convierte dB a ganancia lineal
    y = np.clip(x * g, -MAX_AMPL, MAX_AMPL).astype(np.float32)  # Aplica ganancia y limita
    if to_stereo:  # Si se requiere estéreo
        y = np.column_stack([y, y])  # Duplica L=R
    try:
        sd.play(y, fs, device=device, blocking=block)  # Reproduce con bloqueo opcional
    except sd.PortAudioError as e:  # Maneja errores de dispositivo
        print(f"[Audio] Error de PortAudio: {e}\nDispositivos disponibles:")  # Mensaje de error
        print(sd.query_devices())  # Lista dispositivos
        raise  # Re-lanza la excepción

def ssb_modulate(x, fs, fc, side='USB', fc_error=0.0, phase_error_deg=0.0, carrier_full_scale=0.0):
    """Modula SSB (SC o FC) mediante señal analítica de Hilbert."""
    xa = hilbert(x)  # Señal analítica: x + j*H{x}
    if side.upper() == 'LSB':  # Si se requiere LSB
        xa = np.conj(xa)  # Conjuga para invertir espectro
    fc_eff = fc * (1.0 + fc_error)  # Aplica error relativo de frecuencia
    phi = np.deg2rad(phase_error_deg)  # Convierte error de fase a radianes
    n = np.arange(len(x))  # Índices de muestra
    osc = np.exp(1j * (2*np.pi*fc_eff*n/fs + phi))  # Oscilador complejo e^{j(2πfct+φ)}
    s_sc = np.real(xa * osc)  # Parte real → SSB suprimida de portadora (SC)
    s = s_sc.copy()  # Copia para salida
    if carrier_full_scale > 0.0:  # Si se solicita SSB-FC
        s += carrier_full_scale * np.cos(2*np.pi*fc_eff*n/fs + phi)  # Agrega portadora
    s = np.clip(s, -0.95, 0.95)  # Limita para evitar clipping duro
    return s  # Retorna señal SSB

def make_beep(fs, freq=BEEP_FREQ, ms=BEEP_MS, level=BEEP_LEVEL):
    """Genera un beep senoidal corto a 'freq' Hz, duración 'ms' y nivel 'level'."""
    N = int(fs * (ms / 1000.0))  # Muestras del beep
    t = np.arange(N) / fs  # Vector de tiempo
    env = np.linspace(0, 1, int(0.02*N))  # Envolvente ataque (2%)
    env = np.pad(env, (0, N - len(env)), mode='constant', constant_values=1.0)  # Rellena a 1
    s = level * np.sin(2*np.pi*freq*t) * env  # Senoide con ataque suave
    return s.astype(np.float32)  # Devuelve beep en float32

def make_silence(fs, seconds):
    """Crea silencio de 'seconds' segundos a Fs."""
    N = int(fs * seconds)  # Muestras de silencio
    return np.zeros(N, dtype=np.float32)  # Vector de ceros

def play_triple_beep(fs):
    """Reproduce 3 beeps con separación de 1 segundo entre cada uno."""
    beep = make_beep(fs, BEEP_FREQ, BEEP_MS, BEEP_LEVEL)  # Genera beep
    gap = make_silence(fs, BEEP_GAP_S)  # Genera silencio de 1 s
    seq = np.concatenate([beep, gap, beep, gap, beep, gap])  # Concatena beep-gap-beep-gap-beep
    play_audio(seq, fs, gain_db=0.0, to_stereo=True, device=None, block=True)  # Reproduce secuencia

def play_beep_final(fs):
    """Reproduce 3 beeps con separación de 1 segundo entre cada uno."""
    beep = make_beep(fs, BEEP_FREQ, 600, BEEP_LEVEL)  # Genera beep
    gap = make_silence(fs, BEEP_GAP_S)  # Genera silencio de 1 s
    seq = np.concatenate([gap, beep, gap])  # Concatena beep-gap-beep-gap-beep
    play_audio(seq, fs, gain_db=0.0, to_stereo=True, device=None, block=True)  # Reproduce secuencia


def main():
    wav_path = os.path.join(BASE_DIR, FILENAME)  # Construye ruta completa del WAV
    if not wav_path.lower().endswith(".wav"):
        candidate = wav_path + ".wav"
        if os.path.exists(candidate):
            wav_path = candidate

    if not os.path.exists(wav_path):                                            # Verifica existencia del archivo
        raise FileNotFoundError(f"No se encontró el archivo WAV:\n{wav_path}")  # Error si no existe

    x, fs = read_wav_mono(wav_path)  # Lee WAV mono y obtiene fs

    print("=== WAV CARGADO ===")
    print(f"Ruta: {wav_path}")
    print(f"Fs: {fs} Hz")
    print(f"Muestras: {len(x)}")
    print(f"Duración: {len(x)/fs:.2f} s")
    print(f"pico abs: {np.max(np.abs(x)):.3f}")

    print("\n[SSB] Modulando...")  # Mensaje de estado
    s_ssb = ssb_modulate(
        x, fs, SSB_FC_HZ, side=SSB_SIDE,
        fc_error=SSB_FC_ERROR,
        phase_error_deg=SSB_PHASE_ERROR_DEG,
        carrier_full_scale=SSB_CARRIER_LEVEL
    )

    out_ssb_path = os.path.join(BASE_DIR, "ssb_out.wav")            # Ruta de salida para guardar
    sf.write(out_ssb_path, s_ssb, fs)                               # Guarda WAV modulado
    print(f"[SSB] Señal modulada guardada en: {out_ssb_path}")

    print("\n[TX] Enviando 3 beeps de arranque...")
    play_triple_beep(fs)                                            # Reproduce los 3 beeps (handshake)

    print("[TX] Transmitiendo la señal SSB...")
    play_audio(s_ssb, fs, gain_db=0.0, to_stereo=True, device=None, block=True)  # Reproduce SSB por bocina
    print("[TX] Lista transmision")

    print("[TX] Beep final")
    play_beep_final(fs) 
    print("[TX] Final.")

if __name__ == "__main__":
    main()
