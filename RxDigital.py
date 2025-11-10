#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RxDigital.py — Receptor digital pasobanda (QPSK) con *detección y grabado idénticos* al ReceptorSSB.

Flujo:
 1) Espera EXACTAMENTE 3 beeps @1 kHz (200 ms) con gaps válidos.
 2) Arranca a grabar al 3er beep (arma en 50 ms).
 3) Corta cuando detecta un beep final de 1 kHz sostenido >= 0.60 s,
    con histéresis/decay y respetando un mínimo de grabación (0.60 s).
 4) Luego demodula QPSK y escribe `rx_digital_record.wav` y `rx_payload.bin`.

Esta versión recicla el *estado y umbrales* del ReceptorSSB para el handshake/corte.
"""

import time
import os
import math
import numpy as np
import pyaudio
import soundfile as sf
from dataclasses import dataclass
from enum import Enum, auto
from scipy.signal import butter, sosfiltfilt, sosfilt

# ========================= Estados =========================
class St(Enum):
    WAITING = auto()
    ARMED   = auto()
    RECORD  = auto()
    DONE    = auto()

# ========================= Parámetros (idénticos SSB) =========================
# Señal de beep
BEEP_FREQ        = 1000.0    # Hz
BEEP_MS          = 200.0     # ms
ATTACK_FRAC      = 0.02      # 2% de ataque (plantilla)
SHORT_MIN        = 0.12      # s
SHORT_MAX        = 0.40      # s
GAP_MIN          = 0.60      # s
GAP_MAX          = 1.60      # s
ADAPTIVE_GAP     = True
GAP_TOL_REL      = 0.45      # ±45%
REFRACTORY_S     = 0.22      # s
R_ON             = 0.24      # corr ON
R_OFF            = 0.17      # corr OFF

# --- Corte por beep final ---
END_BEEP_HOLD_S  = 0.60      # 600 ms sostenidos
END_R_ON         = 0.22      # umbral corr baseband
END_TONE_R_ON    = 0.50      # umbral "razón tonal" (Goertzel 1 kHz / energía)
END_DECAY        = 0.5       # tolerancia a microcortes

# Audio/captura
FORCE_FS         = 48000
INPUT_INDEX      = None
BLOCK_S          = 0.020     # 20 ms
MAX_RECORD_S     = 300.0
MIN_REC_S        = 0.60      # s mínimo antes de permitir corte

# Salida
REC_WAV_PATH     = "rx_digital_record.wav"
OUT_BIN_PATH     = "rx_payload.bin"

# ========================= Parámetros QPSK =========================
FC_HZ            = 12000.0   # portadora pasobanda
SYM_RATE         = 2000
RRC_ROLL         = 0.35
RRC_SPAN_SYM     = 8

# ========================= Utilidades SSB (recicladas) ==============
def band_power(x: np.ndarray) -> float:
    return float(np.mean(x*x) + 1e-12)

def goertzel_power(x: np.ndarray, fs: int, f0: float) -> float:
    w = 2.0*np.pi*f0/fs
    c = 2.0*np.cos(w)
    s1 = 0.0; s2 = 0.0
    for xn in x:
        s0 = float(xn) + c*s1 - s2
        s2 = s1
        s1 = s0
    return (s1*s1 + s2*s2 - c*s1*s2) / (len(x) + 1e-12)

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

def bp_tone(fs: int, f0: float, bw: float = 800.0, order: int = 6):
    ny = fs/2.0
    lo_hz = max(100.0, f0 - bw)
    hi_hz = min(ny - 100.0, f0 + bw)
    lo = lo_hz / ny
    hi = max(lo + 0.02, hi_hz / ny)
    hi = min(hi, 0.98)
    return butter(order, [lo, hi], btype='bandpass', output='sos')

# ========================= QPSK helpers ==============================
def rrc_taps(sps, roll, span_sym):
    T = 1.0
    N = int(2*span_sym*sps) + 1
    t = (np.arange(N) - N//2) / sps
    taps = np.zeros_like(t, dtype=np.float64)
    for i, ti in enumerate(t):
        if abs(1 - (4*roll**2)*(ti/T)**2) < 1e-12:
            taps[i] = roll/ (np.sqrt(2)*T) * ((1 + 2/np.pi) * np.sin(np.pi/(4*roll)) + (1 - 2/np.pi) * np.cos(np.pi/(4*roll)))
        elif abs(ti) < 1e-12:
            taps[i] = (1 + roll*(4/np.pi - 1)) / T
        else:
            num = np.sin(np.pi*ti*(1 - roll)/T) + 4*roll*ti*np.cos(np.pi*ti*(1 + roll)/T)/T
            den = np.pi*ti*(1 - (4*roll**2)*(ti/T)**2)/T
            taps[i] = num/den
    taps = taps / np.sqrt(np.sum(taps**2))
    return taps

def mix_down(x, fs, f0):
    n = np.arange(len(x))
    osc = np.exp(-1j*2*np.pi*f0*n/fs)
    return x.astype(np.float64, copy=False) * osc

def estimate_cfo(y):
    ang = np.angle(y[1:] * np.conj(y[:-1]))
    return np.median(ang)

def correct_cfo(y, w):
    n = np.arange(len(y))
    return y * np.exp(-1j*w*n)

def best_offset(mf, sps, search_symbols=400):
    L = min(len(mf), search_symbols*sps)
    e = []
    for off in range(sps):
        seq = mf[off:L:sps]
        e.append(np.sum(np.abs(seq)))
    return int(np.argmax(e))

def slicer_qpsk(z):
    b = []
    for c in z:
        i = c.real; q = c.imag
        bi = 0 if i >= 0 else 1
        bq = 0 if q >= 0 else 1
        g0 = bi
        g1 = bi ^ bq
        b.extend([g0, g1])
    return np.array(b, dtype=np.uint8)

def bits_to_bytes(bits):
    pad = (-len(bits)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    bits = bits.reshape(-1, 8)
    vals = np.packbits(bits, axis=1, bitorder='big').flatten()
    return bytes(vals.tolist())

# ========================= Captura y handshake ========================
def open_stream(pa, fs, input_index):
    block = int(BLOCK_S * fs)
    return pa.open(format=pyaudio.paFloat32, channels=1, rate=fs,
                   input=True, input_device_index=input_index,
                   frames_per_buffer=block)

def record_with_handshake(pa):
    """Replica *idéntica* del flujo de detección/grabado del ReceptorSSB (ajustada mínimo a nombres)."""
    fs = int(FORCE_FS) if FORCE_FS else int(pa.get_default_input_device_info().get('defaultSampleRate', 48000))
    block = int(BLOCK_S * fs)
    templ = make_beep_template(fs)
    templ_N = len(templ)

    # Filtro para ruta 'RF' del beep (alrededor de 1 kHz, como en SSB se pondera por energía)
    sos_bp_beep = bp_tone(fs, BEEP_FREQ, bw=800.0, order=6)

    stream = open_stream(pa, fs, INPUT_INDEX)
    print(f"\n[RX-DIG] Fs={fs} Hz - Buscando 3 beeps @1 kHz ({BEEP_MS:.0f} ms). Deben ser 3: no arma con 2.")

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
        den = float(np.sqrt(np.sum(xb*xb))*1.0)
        if den <= 1e-12: return 0.0
        return num/den

    # Detección de beeps
    r_above = False
    last_cross_end = None
    beeps = []
    first_gap = None

    # Grabación
    rec_chunks = []
    rec_samples = 0
    max_samples = int(MAX_RECORD_S*fs)
    rec_start = None
    bp_hist = []
    hist_len = max(3, int(2.0 / BLOCK_S))  # REC_ADAPT_SEC = 2.0 s

    # Final beep timers
    end_on_time_rf = 0.0
    end_on_time_bb = 0.0
    end_f0 = BEEP_FREQ

    try:
        while True:
            x = np.frombuffer(stream.read(block, exception_on_overflow=False), dtype=np.float32)
            t = now()

            # Correlación normalizada (plantilla 1 kHz con ataque)
            push_block(x)
            r = corr_norm()

            # RUTA RF del beep final (alrededor de 1 kHz)
            x_bp = sosfiltfilt(sos_bp_beep, x.astype(np.float64, copy=False))
            p_bp = band_power(x_bp)

            # Estado: WAITING (3 beeps válidos)
            if state == St.WAITING:
                if r >= R_ON and not r_above:
                    r_above = True
                    last_cross_end = t
                elif r <= R_OFF and r_above:
                    r_above = False
                    beep_start = last_cross_end
                    beep_end = t
                    dur = beep_end - beep_start
                    if SHORT_MIN <= dur <= SHORT_MAX:
                        accept = False
                        if not beeps:
                            beeps = [(beep_start, beep_end)]
                            print("[RX-DIG] Beep 1 detectado.")
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
                                print(f"[RX-DIG] Beep {len(beeps)} OK. gap={gap_ee:.2f} s")
                            else:
                                beeps = [(beep_start, beep_end)]
                                first_gap = None
                                print(f"[RX-DIG] Gap {gap_ee:.2f} s fuera de rango | reinicio.")

                        if len(beeps) == 3 and state == St.WAITING:
                            arm_time = t + 0.05
                            state = St.ARMED
                            print("[RX-DIG] Handshake completado. Armando en 50 ms.")

            if state == St.ARMED and t >= arm_time:
                state = St.RECORD
                rec_start = t
                rec_chunks.clear()
                rec_samples = 0
                bp_hist.clear()
                end_on_time_rf = end_on_time_bb = 0.0
                print(f"[RX-DIG] Grabando… (t={t:.2f} s)")

            # Grabación y corte por beep final
            if state == St.RECORD:
                rec_chunks.append(x.copy())
                rec_samples += len(x)
                if rec_samples >= max_samples:
                    print("[RX-DIG] Máximo de grabación alcanzado. Cortando…")
                    state = St.DONE
                rec_dur = t - rec_start if rec_start is not None else 0.0

                # --- Ruta baseband (correlación 1 kHz crudo) ---
                if r >= END_R_ON:
                    end_on_time_bb += BLOCK_S
                else:
                    end_on_time_bb = max(0.0, end_on_time_bb - END_DECAY*BLOCK_S)

                # --- Ruta RF (tono 1 kHz sobre banda beep) ---
                p_tone = goertzel_power(x_bp, fs, end_f0)
                tone_ratio = p_tone / (p_bp + 1e-12)
                if tone_ratio >= END_TONE_R_ON:
                    end_on_time_rf += BLOCK_S
                else:
                    end_on_time_rf = max(0.0, end_on_time_rf - END_DECAY*BLOCK_S)

                # Cortar cuando cualquiera sostenga >= END_BEEP_HOLD_S y ya se grabó mínimo
                end_on_time = max(end_on_time_bb, end_on_time_rf)
                if end_on_time >= END_BEEP_HOLD_S and rec_dur >= MIN_REC_S:
                    print(f"[RX-DIG] Beep final detectado ({END_BEEP_HOLD_S:.2f} s). Cortando…")
                    state = St.DONE

            if state == St.DONE:
                break

        stream.stop_stream(); stream.close()

        if not rec_chunks:
            print("[RX-DIG] Nada grabado.")
            return False, fs, None

        y = np.concatenate(rec_chunks).astype(np.float64, copy=False)
        sf.write(REC_WAV_PATH, y, fs)
        print(f"[RX-DIG] WAV grabado: {REC_WAV_PATH} ({len(y)/fs:.2f} s)")
        return True, fs, y

    except KeyboardInterrupt:
        print("\n[RX-DIG] Detenido por usuario.")
        try: stream.stop_stream(); stream.close()
        except Exception: pass
        return False, fs, None

# ========================= Demod QPSK ================================
def demodulate_qpsk(y: np.ndarray, fs: int) -> bytes:
    sps = int(round(fs / SYM_RATE))
    # Mezcla a banda base
    ybb = mix_down(y, fs, FC_HZ)
    # LPF
    lp = butter(6, min(0.49, 0.6*SYM_RATE/(fs/2.0)), btype='low', output='sos')
    ybb_i = sosfilt(lp, np.real(ybb))
    ybb_q = sosfilt(lp, np.imag(ybb))
    yc = ybb_i + 1j*ybb_q

    # CFO
    w = estimate_cfo(yc)
    yc = correct_cfo(yc, w)

    # Filtro casado
    rrc = rrc_taps(sps, RRC_ROLL, RRC_SPAN_SYM)
    mf = np.convolve(yc, rrc, mode='same')

    # Muestreo símbolo
    off = best_offset(mf, sps)
    syms = mf[off::sps]
    if np.max(np.abs(syms)) > 0:
        syms = syms / np.median(np.abs(syms) + 1e-12)

    bits = slicer_qpsk(syms)
    # Header opcional de longitud (4 bytes big-endian)
    if len(bits) >= 32:
        hdr = bits_to_bytes(bits[:32])
        L = int.from_bytes(hdr, 'big', signed=False)
        if 0 < L <= (1<<20):
            dbits = bits[32:32+8*L]
            data = bits_to_bytes(dbits)
            if len(data) == L:
                print(f"[RX-DIG] Longitud detectada: {L} bytes")
                return data

    data = bits_to_bytes(bits)
    print(f"[RX-DIG] Sin encabezado válido. Bytes crudos: {len(data)}")
    return data[:65536]

# ========================= Main loop =================================
def run_forever():
    pa = pyaudio.PyAudio()
    try:
        ok, fs, y = record_with_handshake(pa)
        if not ok or y is None:
            return ok, None
        payload = demodulate_qpsk(y, fs)
        if payload:
            with open(OUT_BIN_PATH, "wb") as f:
                f.write(payload)
            print(f"[RX-DIG] Binario escrito: {OUT_BIN_PATH} ({len(payload)} bytes)")
        return True, None
    finally:
        pa.terminate()

if __name__ == "__main__":
    run_forever()
