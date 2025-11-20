#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================
# Receptor digital BPSK con sincronía por beeps
# ------------------------------------------------------------
# - Detecta EXACTAMENTE 3 beeps de 1 kHz (~200 ms c/u) como
#   handshake de inicio.
# - A partir del 3er beep, graba la señal BPSK.
# - Corta ÚNICAMENTE cuando detecta un beep final de 1 kHz
#   sostenido ~1.0 s, usando un detector tonal dedicado.
# - Demodula BPSK coherente (fc = 10 kHz, Rb ~ 5 kbps) para
#   reconstruir un flujo de bits.
# - Empaqueta los bits en bytes (np.packbits) y guarda un
#   archivo con la extensión indicada (jpg, png, wav, pdf, ...).
# ------------------------------------------------------------
# Filosofía:
# - Misma idea que ReceptorSSB: sin timeouts duros.
#   La sesión termina únicamente cuando llega el beep final.
# - Pensado para ser invocado desde menu_receptor.py, pero
#   también funciona ejecutando directamente:
#       python3 RxDigital.py jpg
# ============================================================

import math
import os
import sys
import time
import contextlib

import numpy as np
import pyaudio
import soundfile as sf
from scipy.signal import butter, filtfilt


# ============================================================
# Utilidad para silenciar mensajes ruidosos de ALSA/JACK
# ============================================================

@contextlib.contextmanager
def suppress_alsa_warnings():
    """Silencia mensajes de ALSA/JACK redirigiendo el descriptor de stderr (FD 2)."""
    sys.stderr.flush()
    old_stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(old_stderr_fd, 2)
        os.close(devnull_fd)
        os.close(old_stderr_fd)


# ============================================================
# Parámetros de sincronía por beeps y de audio
# ============================================================

# Tono de los beeps (handshake y final)
BEEP_FREQ       = 1000.0     # Hz
BEEP_MS         = 200.0      # Duración de beeps de handshake (ms)
BEEP_LEVEL      = 0.7        # Amplitud relativa esperada (referencia)

# Detección de 3 beeps de inicio (handshake)
R_ON            = 0.18       # Umbral de correlación "ON" contra plantilla
SHORT_MIN       = 0.12       # Ventana mínima para duración de beep [s]
SHORT_MAX       = 0.45       # Ventana máxima para duración de beep [s]
GAP_MIN         = 0.35       # Mínimo tiempo entre beeps [s]
GAP_MAX         = 2.0        # Máximo tiempo entre beeps [s]
REFRACTORY_S    = 0.18       # Anti-rebote tras un beep detectado [s]

# Detección de beep final (fin de transmisión)
END_BEEP_HOLD_S = 1.00       # Tiempo sostenido requerido para beep final [s]
END_TONE_R_ON   = 0.55       # Umbral sobre razón "potencia_tonal / potencia_total"
END_DECAY       = 0.5        # Tolerancia a microcortes (decaimiento progresivo)

# Audio / PyAudio
FORCE_FS        = 48000      # Hz (frecuencia de muestreo de trabajo)
INPUT_INDEX     = None       # None => dispositivo de entrada por defecto
BLOCK_S         = 0.020      # Duración de bloque de captura [s] (~20 ms)

# Parámetros de la BPSK (deben coincidir con el TX)
BPSK_FC         = 10_000.0   # Hz, frecuencia de la portadora BPSK
BPSK_BIT_RATE   = 5_000.0    # bits/s


# ============================================================
# Plantilla de beep y correlación (para handshake)
# ============================================================

def make_beep_template(fs: int) -> np.ndarray:
    """Genera una plantilla normalizada del beep de 1 kHz (~BEEP_MS ms).

    Esta plantilla se utiliza para hacer correlación normalizada
    sobre el bloque más reciente de audio, lo cual permite detectar
    la presencia de cada beep de handshake.
    """
    dur = BEEP_MS / 1000.0
    N = int(round(dur * fs))
    t = np.arange(N) / fs

    beep = np.sin(2.0 * np.pi * BEEP_FREQ * t) * BEEP_LEVEL
    win = np.hanning(N)
    beep = (beep * win).astype(np.float64)

    # Normalización de energía para que la correlación sea comparable
    norm = np.sqrt(np.sum(beep ** 2) + 1e-12)
    return beep / norm


def corrcoef_template(x: np.ndarray, templ: np.ndarray) -> float:
    """Correlación normalizada entre el final de x y la plantilla templ.

    Implementa algo similar a un coeficiente de correlación de Pearson
    entre la ventana más reciente de x y la plantilla de beep. Se
    utiliza principalmente para detectar los 3 beeps de handshake.
    """
    Nx = len(x)
    Nt = len(templ)
    if Nx < Nt or Nt == 0:
        return 0.0

    seg = x[-Nt:].astype(np.float64)
    seg = seg - np.mean(seg)
    templ_z = templ - np.mean(templ)

    num = np.sum(seg * templ_z)
    den = np.sqrt(np.sum(seg ** 2) * np.sum(templ_z ** 2) + 1e-20)
    if den <= 0.0:
        return 0.0
    return float(num / den)


# ============================================================
# Detector tonal (Goertzel) para beep final
# ============================================================

def goertzel_power(x: np.ndarray, fs: float, f0: float) -> float:
    """Calcula la potencia de un tono f0 en x usando el algoritmo de Goertzel.

    Esta función permite estimar cuánta energía hay concentrada en
    torno a f0 (en este caso, 1 kHz) dentro de un bloque de audio,
    sin necesidad de calcular un espectro completo.
    """
    N = len(x)
    if N <= 0:
        return 0.0

    k = int(0.5 + (N * f0) / fs)
    w = 2.0 * math.pi * k / N
    cos_w = math.cos(w)
    coeff = 2.0 * cos_w

    s_prev = 0.0
    s_prev2 = 0.0
    for n in range(N):
        s = float(x[n]) + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s

    power = s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2
    return float(max(power, 0.0)) / max(N, 1)


# ============================================================
# Demodulación BPSK coherente
# ============================================================

def bpsk_demodulate(y: np.ndarray, fs: float = 48000.0,
                    fc: float = 10_000.0,
                    bit_rate: float = 5_000.0) -> np.ndarray:
    """Demodulación BPSK coherente simple.

    Pasos:
    1) Mezcla la señal con cos(2π f_c t) para obtener la rama I.
    2) Aplica un filtro pasa-bajo a la rama I (ancho ≈ 0.75 * bit_rate).
    3) Aplica una operación "integrate & dump" sobre ventanas de
       tamaño aproximado a "muestras por bit" para decidir el signo
       de cada símbolo (bit).
    """
    if len(y) == 0:
        return np.array([], dtype=np.uint8)

    t = np.arange(len(y)) / float(fs)
    i_bb = y * np.cos(2.0 * np.pi * fc * t)

    # Filtro pasa-bajo para limpiar la rama I en baseband
    cutoff = min(0.75 * bit_rate, 0.45 * fs)
    b, a = butter(4, cutoff / (0.5 * fs))
    i_f = filtfilt(b, a, i_bb)

    # Muestras por bit (aprox)
    spb = max(1, int(round(fs / bit_rate)))
    n_sym = len(i_f) // spb
    if n_sym <= 0:
        return np.array([], dtype=np.uint8)

    # Integrate & dump: promedio por símbolo
    acc = i_f[:n_sym * spb].reshape(n_sym, spb).mean(axis=1)
    bits = (acc > 0).astype(np.uint8)
    return bits


# ============================================================
# Reconstrucción de archivo a partir de bits
# ============================================================

def save_bits_as_file(bits: np.ndarray, filename: str) -> None:
    """Empaqueta bits (0/1) en bytes y escribe archivo binario.

    Esta función realiza la operación inversa a np.unpackbits, la cual
    es la utilizada en el transmisor para convertir un archivo arbitrario
    en una secuencia de bits a transmitir.
    """
    data = np.packbits(bits)
    with open(filename, "wb") as f:
        f.write(data)
    print(f"[RX]: Archivo reconstruido: {filename} ({len(data)} bytes)")


# ============================================================
# Máquina de estados de recepción
# ------------------------------------------------------------
# WAITING : Buscando el patrón de 3 beeps iniciales.
# ARMED   : Handshake completo, listo para registrar.
# RECORD  : Grabando la señal BPSK y vigilando beep final.
# DONE    : Se detectó beep final; se corta la sesión.
# ============================================================

class St:
    WAITING = 0
    ARMED   = 1
    RECORD  = 2
    DONE    = 3


# ============================================================
# Función principal de una sesión de recepción
# ============================================================

def run_one_session(pa: pyaudio.PyAudio, outfile_ext: str = ".bin") -> bool:
    """Ejecuta una sesión de recepción BPSK.

    Flujo:
    1) Abre un stream de audio de entrada.
    2) Espera 3 beeps de 1 kHz (handshake).
    3) A partir de ahí, entra en modo RECORD y acumula bloques.
    4) Supervisa un beep final de 1 kHz sostenido (~1 s) usando Goertzel.
    5) Demodula y guarda el archivo reconstruido con extensión `outfile_ext`.
    """
    # Frecuencia de muestreo y tamaño de bloque
    fs = int(FORCE_FS) if FORCE_FS else int(pa.get_default_input_device_info().get("defaultSampleRate", 48000))
    block = int(BLOCK_S * fs)

    # Plantilla del beep de 1 kHz para correlación de handshake
    templ = make_beep_template(fs)
    templ_N = len(templ)

    print(f"\n[RX]: Fs={fs} Hz - Esperando 3 beeps de 1 kHz (handshake).")

    # Apertura del stream de entrada con ALSA/JACK silenciados
    with suppress_alsa_warnings():
        stream = pa.open(format=pyaudio.paFloat32,
                         channels=1,
                         rate=fs,
                         input=True,
                         input_device_index=INPUT_INDEX,
                         frames_per_buffer=block)

    # Reloj relativo
    t0 = time.time()
    now = lambda: time.time() - t0

    # Buffer deslizante para correlación
    buf = np.zeros(templ_N, dtype=np.float64)

    def push_block(xblock: np.ndarray) -> np.ndarray:
        """Desplaza el buffer 'buf' e inserta el nuevo bloque al final."""
        nonlocal buf
        if len(xblock) == 0:
            return buf
        xblock = xblock.astype(np.float64, copy=False)
        if len(xblock) >= templ_N:
            buf = xblock[-templ_N:].copy()
        else:
            shift = len(xblock)
            buf = np.roll(buf, -shift)
            buf[-shift:] = xblock
        return buf

    # Inicialización de la máquina de estados
    state = St.WAITING

    # Variables para detección de beeps de handshake
    beeps = []              # lista de (t_inicio, t_fin) de cada beep válido
    last_beep_end = None    # tiempo de fin del último beep aceptado
    last_cross_on = None    # instante donde la correlación cruza R_ON al alza
    refractory_until = 0.0  # ventana anti-rebote

    # Variables de grabación y beep final
    rec_chunks   = []   # bloques de audio acumulados en RECORD
    rec_samples  = 0    # contador total de muestras grabadas
    end_beep_hold = 0.0 # tiempo acumulado por el beep final

    try:
        while True:
            # Lectura de un bloque de audio
            data = stream.read(block, exception_on_overflow=False)
            x = np.frombuffer(data, dtype=np.float32)
            t = now()

            # Actualizar buffer de correlación
            buf = push_block(x)
            r = corrcoef_template(buf, templ)

            # ------------------------------------------------------
            # Estado WAITING: búsqueda de 3 beeps de handshake
            # ------------------------------------------------------
            if state == St.WAITING:
                if r >= R_ON:
                    # Se supera el umbral: posible inicio de beep
                    if last_cross_on is None:
                        last_cross_on = t
                else:
                    # r < R_ON : el beep terminó (si había uno en curso)
                    if last_cross_on is not None:
                        dur = t - last_cross_on
                        last_cross_on = None

                        # Validación de duración del beep
                        if dur >= SHORT_MIN and dur <= SHORT_MAX and t >= refractory_until:
                            if last_beep_end is None:
                                # Primer beep aceptado
                                beeps = [(t - dur, t)]
                                print("[RX]: Beep 1 detectado.")
                            else:
                                # Hay beeps previos: verificar el gap
                                gap = (t - dur) - last_beep_end
                                if GAP_MIN <= gap <= GAP_MAX:
                                    beeps.append((t - dur, t))
                                    print(f"[RX]: Beep {len(beeps)} detectado (gap={gap:.2f} s).")
                                else:
                                    # Patrón inválido, se reinicia conteo
                                    beeps = [(t - dur, t)]
                                    print(f"[RX]: Gap {gap:.2f} s fuera de rango; reiniciando patrón.")
                            last_beep_end = t
                            refractory_until = t + REFRACTORY_S

                            # Cuando se acumulan 3 beeps válidos, se arma
                            if len(beeps) == 3:
                                print("[RX]: Handshake completado. Iniciando grabación.")
                                state = St.ARMED
                                rec_chunks.clear()
                                rec_samples = 0
                                end_beep_hold = 0.0
                                continue

            # ------------------------------------------------------
            # Estado ARMED: primer bloque tras el handshake
            # ------------------------------------------------------
            if state == St.ARMED:
                # Pasamos inmediatamente a RECORD para acumular datos
                state = St.RECORD
                print("[RX]: Grabando señal BPSK...")
                # No hacemos "continue" para que este mismo bloque
                # también se procese como parte de RECORD.

            # ------------------------------------------------------
            # Estado RECORD: acumulación y detección de beep final
            # ------------------------------------------------------
            if state == St.RECORD:
                # 1) Acumular el audio en el buffer de grabación
                rec_chunks.append(x.copy())
                rec_samples += len(x)

                # 2) Detector dedicado para beep final
                #    - Calculamos energía total aproximada del bloque.
                #    - Calculamos energía tonal Goertzel en BEEP_FREQ.
                #    - Evaluamos la razón tonal vs total.
                p_total = float(np.mean(x.astype(np.float64) ** 2))
                p_tone  = goertzel_power(x, fs, BEEP_FREQ)
                tone_ratio = p_tone / (p_total + 1e-12)

                if tone_ratio >= END_TONE_R_ON:
                    # El bloque está dominado por un tono cercano a 1 kHz
                    end_beep_hold += BLOCK_S
                else:
                    # Se reduce progresivamente el tiempo acumulado
                    end_beep_hold = max(0.0, end_beep_hold - END_DECAY * BLOCK_S)

                # 3) Condición de corte por beep final sostenido
                if end_beep_hold >= END_BEEP_HOLD_S:
                    print(f"[RX]: Beep final detectado ({END_BEEP_HOLD_S:.2f} s). Deteniendo sesión.")
                    state = St.DONE

            # ------------------------------------------------------
            # Estado DONE: salir del bucle principal
            # ------------------------------------------------------
            if state == St.DONE:
                break

    finally:
        # Cierre ordenado del stream
        stream.stop_stream()
        stream.close()

    # Si no se grabó nada útil, abortar la sesión
    if rec_samples == 0:
        print("[RX]: No se capturó señal útil.")
        return False

    # Concatenar todos los bloques grabados
    rec = np.concatenate(rec_chunks) if rec_chunks else np.zeros(0, dtype=np.float32)
    print(f"[RX]: Muestras capturadas: {len(rec)} ({len(rec) / fs:.2f} s).")

    # Guardar WAV de referencia para posibles depuraciones
    try:
        sf.write("captura_bpsk_digital.wav", rec, fs)
        print("[RX]: Captura guardada en captura_bpsk_digital.wav.")
    except Exception as e:
        print(f"[RX]: No fue posible guardar la captura WAV: {e}")

    # Demodulación BPSK coherente
    bits = bpsk_demodulate(rec, fs=fs, fc=BPSK_FC, bit_rate=BPSK_BIT_RATE)
    print(f"[RX]: Bits demodulados: {len(bits)}.")

    # Reconstrucción de archivo recibido
    save_bits_as_file(bits, "archivo_recibido" + outfile_ext)
    print("[RX]: Sesión de recepción digital finalizada correctamente.")
    return True


# ============================================================
# Función main: lectura de argumentos y bucle de sesiones
# ============================================================

def main() -> None:
    """Punto de entrada del receptor digital BPSK."""
    # Extensión de archivo (primer argumento)
    if len(sys.argv) > 1:
        raw = sys.argv[1].strip()
        if raw.startswith("."):
            ext = raw
        else:
            ext = "." + raw
    else:
        ext = ".bin"

    print(f"[RX]: Extensión de archivo de salida seleccionada: {ext}")

    # Inicialización de PyAudio con warnings de ALSA/JACK silenciados
    with suppress_alsa_warnings():
        pa = pyaudio.PyAudio()

    try:
        while True:
            ok = run_one_session(pa, outfile_ext=ext)
            if not ok:
                print("[RX]: Sesión con fallo; en espera de la siguiente transmisión.")
            # Si se desea finalizar después de una sola sesión, descomentar:
            # break
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n[RX]: Receptor digital detenido por el usuario.")
    finally:
        pa.terminate()


if __name__ == "__main__":
    main()
