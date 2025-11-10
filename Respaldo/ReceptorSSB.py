#!/usr/bin/env python3

# ====================================================
# Receptor SSB con sincronía por 3 beeps + corte por beep final
# ----------------------------------------------------
# - Detecta EXACTAMENTE 3 beeps (no arma con 1 ni con 2)
# - Arranca a grabar al 3er beep
# - Corta cuando detecta un beep final de 1 kHz sostenido 600 ms
#   (y opcionalmente por baja energía si activas el fallback)
# - Demodula SSB-USB (producto + LPF) y guarda WAV fijo: "Demodulado.wav"
#------------------------------------------------------
# - Para ejectuar programa: "python3 ReceptorSSB.py"
# - Para reproducir .wav: aplay Demodulado.wav
# ====================================================

import time
import numpy as np
import pyaudio
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

#====================================
# PARÁMETROS DEL SISTEMA
#------------------------------------
# Ajustes de sincronía (beeps), armado por timeout, filtros SSB,
# criterios de corte (beep final y/o energía), audio y salida.
#====================================

# Beeps del TX
BEEP_FREQ        = 1000.0   # Hz
BEEP_MS          = 200.0    # ms
ATTACK_FRAC      = 0.02     # 2 % de ataque
SHORT_MIN        = 0.12     # s
SHORT_MAX        = 0.40     # s
GAP_MIN          = 0.60     # s
GAP_MAX          = 1.60     # s
ADAPTIVE_GAP     = True
GAP_TOL_REL      = 0.45     # ±45 %
REFRACTORY_S     = 0.22     # s
R_ON             = 0.24     # umbral ON de correlación normalizada
R_OFF            = 0.17     # umbral OFF

# Timeout
T_ARM_TIMEOUT0   = 8.0      # s si no hay beeps aún
T_ARM_EXTEND     = 6.0      # s adicionales si ya hubo 1–2 beeps

# SSB
SSB_FC_HZ        = 12000.0  # Hz
BASEBAND_CUT     = 4500.0   # Hz
BP_BW_HZ         = 4000.0   # Hz
BP_ORDER         = 6

# Corte automático por energía
USE_FALLBACK_ENERGY = False
STOP_HOLD_S      = 0.80     # s bajo umbral para cortar
REC_ADAPT_SEC    = 2.0      # s de ventana para baseline durante grava
K_OFF_BP         = 3.0
MIN_REC_S        = 0.60     # s mínimo antes de permitir corte

# --- Beep final ---
END_BEEP_HOLD_S  = 0.60     # 600 ms sostenidos
END_R_ON         = 0.22     # umbral correlación baseband 
END_TONE_R_ON    = 0.50     # umbral razón tonal en RF 
END_DECAY        = 0.5      # tolerancia a microcortes

# Audio
FORCE_FS         = 48000
INPUT_INDEX      = None
BLOCK_S          = 0.020    # 20 ms
MAX_RECORD_S     = 300.0

# Salida
OUT_PREFIX       = "Demodulado"
PLAY_OUTPUT      = True
DEBUG            = False
DBG_EVERY_S      = 0.3

#====================================
# PLANTILLAS Y FILTROS
#------------------------------------
# - make_beep_template: plantilla normalizada del beep de 1 kHz.
# - bp_ssb: banda alrededor de la portadora.
# - lpf_baseband: pasa-bajo para recuperar banda base tras producto.
#====================================

def make_beep_template(fs: int) -> np.ndarray:
    N = int(round(fs * (BEEP_MS/1000.0)))
    t = np.arange(N) / fs
    env = np.ones(N, dtype=np.float64)
    atk = max(1, int(round(N*ATTACK_FRAC)))
    env[:atk] = np.linspace(0.0, 1.0, atk, dtype=np.float64)
    s = np.sin(2*np.pi*BEEP_FREQ*t) * env
    s = s - np.mean(s)
    s /= (np.sqrt(np.sum(s*s)) + 1e-12)
    return s

def bp_ssb(fs: int, fc: float, bw: float, order: int = 6):
    ny = fs/2.0
    lo_hz = max(300.0, fc - bw)
    hi_hz = min(ny - 500.0, fc + bw)
    lo = lo_hz / ny
    hi = max(lo + 0.05, hi_hz / ny)
    hi = min(hi, 0.98)
    return butter(order, [lo, hi], btype='bandpass', output='sos')

def lpf_baseband(fs: int, fcut: float, order: int = 6):
    return butter(order, fcut/(fs/2), btype='low', output='sos')

#====================================
# MEDIDAS DE POTENCIA Y DETECCIÓN
#------------------------------------
# - band_power: potencia media.
# - goertzel_power: potencia a f0 con Goertzel.
#====================================

def band_power(x: np.ndarray) -> float:
    return float(np.mean(x*x) + 1e-12)

def goertzel_power(x: np.ndarray, fs: int, f0: float) -> float:
    """Potencia a f0 (Hz) con Goertzel sobre el bloque x."""
    w = 2.0*np.pi*f0/fs
    c = 2.0*np.cos(w)
    s1 = 0.0
    s2 = 0.0
    for xn in x:
        s0 = float(xn) + c*s1 - s2
        s2 = s1
        s1 = s0
    return (s1*s1 + s2*s2 - c*s1*s2) / (len(x) + 1e-12)

#====================================
# DEMODULACIÓN SSB
#------------------------------------
# - ssb_demod_product: demod. por producto con coseno a fc y LPF.
#   Normaliza la salida para evitar saturación al guardar WAV.
#====================================

def ssb_demod_product(sig: np.ndarray, fs: int, fc: float, phase_deg: float = 0.0):
    n = np.arange(len(sig))
    phi = np.deg2rad(phase_deg)
    lo = 2.0*np.cos(2*np.pi*fc*n/fs + phi)
    y = sig * lo
    y = sosfiltfilt(lpf_baseband(fs, BASEBAND_CUT), y)
    peak = np.max(np.abs(y)) or 1.0
    return (y/peak)*0.9

#====================================
# MÁQUINA DE ESTADOS
#------------------------------------
# - St: estados del RX (WAITING→ARMED→RECORD→DONE).
#====================================

class St:
    WAITING, ARMED, RECORD, DONE = range(4)

#====================================
# CORE DEL RECEPTOR
#------------------------------------
# - run_one_session: ciclo completo:
#   (1) Espera patrón de 3 beeps.
#   (2) Arma y graba,
#   (3) Corta por beep final (1 kHz por 600 ms),
#   (4) Demodula y guarda "Demodulado.wav".
#====================================

def run_one_session(pa: pyaudio.PyAudio):
    fs = int(FORCE_FS) if FORCE_FS else int(pa.get_default_input_device_info().get('defaultSampleRate', 48000))
    block = int(BLOCK_S * fs)
    templ = make_beep_template(fs)
    templ_N = len(templ)
    sos_bp = bp_ssb(fs, SSB_FC_HZ, BP_BW_HZ, BP_ORDER)

    stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=fs, input=True,
                     input_device_index=INPUT_INDEX, frames_per_buffer=block)
    print(f"\n[RX] Fs={fs} Hz - Buscando 3 beeps @1 kHz ({BEEP_MS:.0f} ms). Deben ser 3: no arma con 2.")

    state = St.WAITING
    t0 = time.time(); now = lambda: time.time()-t0

    buf = np.zeros(templ_N, dtype=np.float64)
    def push_block(xblock):
        nonlocal buf
        x = xblock.astype(np.float64, copy=False)
        m = len(x)
        if m >= len(buf):
            buf = x[-len(buf):].copy()
        else:
            buf = np.concatenate([buf[m:], x])

    def corr_norm():
        xb = buf - np.mean(buf)
        num = float(np.dot(xb, templ))
        den = float(np.sqrt(np.sum(xb*xb))*1.0)  # templ unidad
        if den <= 1e-12: return 0.0
        return num/den

    # Detección de beeps
    r_above = False
    last_cross_end = None
    beeps = []          # lista de (t_start, t_end)
    first_gap = None
    arm_deadline = now() + T_ARM_TIMEOUT0
    arm_extended_to = None

    # Grabación
    rec_chunks = []
    rec_samples = 0
    max_samples = int(MAX_RECORD_S*fs)
    rec_start = None
    stop_below_since = None
    bp_hist = []  # historia de p_bp durante grabación (para baseline real)
    hist_len = max(3, int(REC_ADAPT_SEC / BLOCK_S))

    # --- Acumuladores de tono final ---
    end_on_time = 0.0
    end_on_time_rf = 0.0  # ruta RF (modulado)
    end_on_time_bb = 0.0  # ruta baseband (1 kHz crudo)
    end_f0 = SSB_FC_HZ + BEEP_FREQ  # ~13 kHz si BEEP_FREQ=1 kHz
    last_dbg = time.time()

    try:
        while True:
            data = stream.read(block, exception_on_overflow=False)
            x = np.frombuffer(data, dtype=np.float32)
            t = now()
            push_block(x)
            r = corr_norm()

            if DEBUG and (time.time()-last_dbg) >= DBG_EVERY_S:
                print(f"[DBG] t={t:6.2f}s r={r:0.3f} beeps={len(beeps)} st={state}")
                last_dbg = time.time()

            if state == St.WAITING:
                if not r_above and r >= R_ON:
                    if last_cross_end is None or (t - last_cross_end) >= REFRACTORY_S:
                        r_above = True
                        beep_start = t
                elif r_above and r <= R_OFF:
                    r_above = False
                    beep_end = t
                    last_cross_end = t
                    dur = beep_end - beep_start

                    if SHORT_MIN <= dur <= SHORT_MAX:
                        accept = False
                        if not beeps:
                            beeps = [(beep_start, beep_end)]
                            print("[RX] Beep 1 detectado (corr).")
                        else:
                            prev_start, prev_end = beeps[-1]
                            gap_ee = beep_end - prev_end
                            if GAP_MIN <= gap_ee <= GAP_MAX:
                                accept = True
                            if ADAPTIVE_GAP:
                                if first_gap is None and len(beeps) == 1:
                                    first_gap = gap_ee
                                elif first_gap is not None:
                                    low = first_gap*(1.0 - GAP_TOL_REL)
                                    high = first_gap*(1.0 + GAP_TOL_REL)
                                    if low <= gap_ee <= high:
                                        accept = True
                            if accept:
                                beeps.append((beep_start, beep_end))
                                print(f"[RX] Beep {len(beeps)} detectado (gap={gap_ee:.2f} s).")
                            else:
                                beeps = [(beep_start, beep_end)]
                                first_gap = None
                                print(f"[RX] Gap {gap_ee:.2f} s fuera de rango → reinicio suave.")

                        if len(beeps) == 3 and state == St.WAITING:
                            arm_time = t + 0.05
                            state = St.ARMED
                            print("[RX] Patrón OK (3 beeps). Armando en 50 ms.")

            # Timeout de armado
            if state == St.WAITING:
                if len(beeps) == 0 and t >= arm_deadline:
                    state = St.ARMED
                    arm_time = t + 0.05
                    print("[RX] Timeout sin beeps. Armando por energía en banda.")
                elif 1 <= len(beeps) <= 2:
                    if arm_extended_to is None:
                        arm_extended_to = t + T_ARM_EXTEND
                    elif t >= arm_extended_to:
                        beeps = []
                        first_gap = None
                        arm_deadline = t + T_ARM_TIMEOUT0
                        arm_extended_to = None
                        print("[RX] No llegaron 3 beeps a tiempo → reinicio de patrón y espera.")

            # Transición a grabación
            if state == St.ARMED and t >= arm_time:
                state = St.RECORD
                rec_start = t
                rec_chunks.clear()
                rec_samples = 0
                stop_below_since = None
                bp_hist.clear()
                end_on_time = end_on_time_rf = end_on_time_bb = 0.0
                print(f"[RX] Grabando… (t={t:.2f} s)")

            # Band-pass y potencia para ruta RF
            x_bp = sosfiltfilt(sos_bp, x.astype(np.float64, copy=False))
            p_bp = band_power(x_bp)

            # Grabación
            if state == St.RECORD:
                rec_chunks.append(x.copy())
                rec_samples += len(x)
                rec_dur = t - rec_start if rec_start is not None else 0.0

                # --- Ruta baseband (correlación 1 kHz crudo) ---
                if r >= END_R_ON:
                    end_on_time_bb += BLOCK_S
                else:
                    end_on_time_bb = max(0.0, end_on_time_bb - END_DECAY*BLOCK_S)

                # --- Ruta RF (Goertzel @ fc + 1 kHz) ---
                p_tone = goertzel_power(x_bp, fs, end_f0)
                tone_ratio = p_tone / (p_bp + 1e-12)
                if tone_ratio >= END_TONE_R_ON:
                    end_on_time_rf += BLOCK_S
                else:
                    end_on_time_rf = max(0.0, end_on_time_rf - END_DECAY*BLOCK_S)

                # Cortar cuando cualquiera sostenga 600 ms
                end_on_time = max(end_on_time_bb, end_on_time_rf)
                if end_on_time >= END_BEEP_HOLD_S and rec_dur >= MIN_REC_S:
                    print(f"[RX] Beep final detectado ({END_BEEP_HOLD_S:.2f} s). Cortando…")
                    state = St.DONE

                # Fallback por energía (opcional)
                if USE_FALLBACK_ENERGY and state == St.RECORD:
                    bp_hist.append(p_bp)
                    if len(bp_hist) > hist_len:
                        bp_hist.pop(0)

                    if rec_dur >= MIN_REC_S and len(bp_hist) >= 5:
                        arr = np.array(bp_hist, dtype=np.float64)
                        med = float(np.median(arr))
                        mad = float(np.median(np.abs(arr - med)) + 1e-12)
                        thr_off = med + K_OFF_BP * mad
                        if p_bp < thr_off:
                            if stop_below_since is None:
                                stop_below_since = t
                            elif (t - stop_below_since) >= STOP_HOLD_S:
                                print(f"[RX] Energía banda baja {STOP_HOLD_S:.2f} s. Cortando…")
                                state = St.DONE
                        else:
                            stop_below_since = None

                if rec_samples >= max_samples and state == St.RECORD:
                    print("[RX] Límite de grabación alcanzado.")
                    state = St.DONE

            if state == St.DONE:
                break

    finally:
        stream.stop_stream()
        stream.close()

    if rec_samples == 0:
        print("[RX] No se capturó señal útil.")
        return False, fs

    # Demodulación y guardado
    rec = np.concatenate(rec_chunks) if rec_chunks else np.zeros(0, dtype=np.float32)
    print(f"[RX] Capturadas {len(rec)} muestras ({len(rec)/fs:.2f} s). Demodulando…")

    x_hat = ssb_demod_product(sosfiltfilt(sos_bp, rec.astype(np.float64, copy=False)), fs, SSB_FC_HZ, phase_deg=0.0)
    out = (x_hat / (np.max(np.abs(x_hat)) or 1.0) * 0.9).astype(np.float32)

    fname = "Demodulado.wav"
    sf.write(fname, out, fs)
    print(f"[RX] WAV guardado: {fname}")

    if PLAY_OUTPUT:
        try:
            out_stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=fs, output=True)
            out_stream.write(out.tobytes())
            out_stream.stop_stream()
            out_stream.close()
            print("[RX] Reproducción OK.")
        except Exception as e:
            print(f"[RX] No pude reproducir: {e}")

    print("[RX] Sesión OK.")
    return True, fs

#====================================
# BUCLE PRINCIPAL
#====================================

def run_forever():
    pa = pyaudio.PyAudio()
    try:
        print("[RX] Receptor continuo iniciado.")
        while True:
            try:
                ok, _ = run_one_session(pa)
                time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n[RX] Detenido por usuario.")
                break
            except Exception as e:
                print(f"[RX] Error: {e}. Reinicio…")
                time.sleep(0.5)
    finally:
        pa.terminate()

#====================================
# Main
#====================================

if __name__ == "__main__":
    run_forever()

