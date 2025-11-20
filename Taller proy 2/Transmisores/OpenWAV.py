#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===============================================================
# TRANSMISOR SSB / ISB – CLI + defaults (sin rutas hardcode)
# ---------------------------------------------------------------
# - Acepta argumentos:
#     --mode SSB|ISB
#     --wav1 <path> [--wav2 <path> para ISB]
#     --side USB|LSB (solo SSB)
#     --fc <Hz> [--phase <deg>] [--freqerr <rel>] [--carrier <0..1>]
#     --tx 0|1 [--handshake 0|1] [--gaindb <dB>]
#     --save <path> (opcional; si no, guarda en el mismo folder del WAV1)
# - Reproduce por altavoz (sounddevice); si falla, intenta PyAudio.
# - Handshake: 3 beeps de 1 kHz y beep final (600 ms).
# ===============================================================

import argparse
import os
import sys
import numpy as np
import soundfile as sf
from scipy.signal import hilbert

# ---------- Audio backends ----------
_BACKENDS = {}

def _backend_sd_play(y, fs):
    import sounddevice as sd
    sd.play(y, fs, blocking=True)

def _backend_pyaudio_play(y, fs):
    import pyaudio
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=pyaudio.paFloat32, channels=y.shape[1] if y.ndim==2 else 1,
                        rate=fs, output=True)
        stream.write(y.astype(np.float32).tobytes())
        stream.stop_stream(); stream.close()
    finally:
        p.terminate()

def play_audio(x, fs, gain_db=0.0, stereo=True):
    g = 10.0 ** (gain_db / 20.0)
    y = np.clip(x * g, -0.95, 0.95).astype(np.float32)
    if stereo and y.ndim == 1:
        y = np.column_stack([y, y])
    # Try sounddevice then PyAudio
    try:
        _backend_sd_play(y, fs)
        return
    except Exception as e_sd:
        try:
            _backend_pyaudio_play(y, fs)
            return
        except Exception as e_pa:
            print(f"[Audio] No se pudo reproducir: {e_sd} | {e_pa}")

# ---------- Beeps ----------
BEEP_FREQ = 1000.0
BEEP_MS   = 200.0
BEEP_GAP_S= 1.0
BEEP_LEVEL= 0.6

def make_beep(fs, freq=BEEP_FREQ, ms=BEEP_MS, level=BEEP_LEVEL):
    N = int(fs * (ms/1000.0))
    t = np.arange(N)/fs
    atk = max(1, int(0.02*N))
    env = np.ones(N, dtype=np.float64)
    env[:atk] = np.linspace(0.0, 1.0, atk)
    s = level * np.sin(2*np.pi*freq*t) * env
    return s.astype(np.float32)

def make_silence(fs, seconds):
    return np.zeros(int(fs*seconds), dtype=np.float32)

def play_triple_beep(fs):
    seq = np.concatenate([make_beep(fs), make_silence(fs, BEEP_GAP_S),
                          make_beep(fs), make_silence(fs, BEEP_GAP_S),
                          make_beep(fs)])
    play_audio(seq, fs, gain_db=0.0, stereo=True)

def play_beep_final(fs):
    seq = np.concatenate([make_silence(fs, BEEP_GAP_S),
                          make_beep(fs, ms=600),
                          make_silence(fs, BEEP_GAP_S)])
    play_audio(seq, fs, gain_db=0.0, stereo=True)

# ---------- WAV I/O ----------

def read_wav_mono(path):
    if not path.lower().endswith(".wav"):
        raise ValueError("La ruta debe terminar en .wav (no se agrega automático).")
    x, fs = sf.read(path, always_2d=True)
    x = x.mean(axis=1).astype(np.float64)
    peak = float(np.max(np.abs(x))) if x.size else 1.0
    if peak > 1.0:
        x = x / peak
    return x, fs

def linear_resample(x, fs_in, fs_out):
    if fs_in == fs_out:
        return x, fs_in
    # Resample by simple linear interpolation
    ratio = fs_out / fs_in
    n_out = int(round(len(x) * ratio))
    t_in = np.linspace(0.0, len(x)/fs_in, num=len(x), endpoint=False)
    t_out= np.linspace(0.0, len(x)/fs_in, num=n_out,   endpoint=False)
    y = np.interp(t_out, t_in, x).astype(np.float64)
    return y, fs_out

# ---------- Modulación ----------

def ssb_modulate(x, fs, fc, side="USB", fc_error=0.0, phase_error_deg=0.0, carrier_full_scale=0.0):
    xa = hilbert(x)
    if side.upper() == "LSB":
        xa = np.conj(xa)
    fc_eff = fc * (1.0 + fc_error)
    phi = np.deg2rad(phase_error_deg)
    n = np.arange(len(x))
    osc = np.exp(1j * (2*np.pi*fc_eff*n/fs + phi))
    s_sc = np.real(xa * osc)
    s = s_sc.copy()
    if carrier_full_scale > 0.0:
        s += carrier_full_scale * np.cos(2*np.pi*fc_eff*n/fs + phi)
    return np.clip(s, -0.95, 0.95).astype(np.float32)

# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description="TX SSB/ISB por audio (beeps + señal + beep final)")
    p.add_argument("--mode", choices=["SSB","ISB"], default="SSB")
    p.add_argument("--wav1", required=True, help="Ruta WAV 1 (obligatorio, con .wav)")
    p.add_argument("--wav2", help="Ruta WAV 2 (solo ISB, con .wav)")

    p.add_argument("--side", choices=["USB","LSB"], default="USB", help="Solo SSB")
    p.add_argument("--fc", type=float, default=12000.0)
    p.add_argument("--phase", type=float, default=0.0)
    p.add_argument("--freqerr", type=float, default=0.0, help="Relativo, ej. 0.02 = +2%")
    p.add_argument("--carrier", type=float, default=0.0, help="Nivel de portadora (0..1). 0.0 = SC")

    p.add_argument("--tx", type=int, default=1, help="1=Transmitir por altavoz; 0=solo generar WAV")
    p.add_argument("--handshake", type=int, default=1, help="1=Enviar 3 beeps de arranque y beep final")
    p.add_argument("--gaindb", type=float, default=0.0, help="Ganancia TX en dB")
    p.add_argument("--save", type=str, default="", help="Ruta WAV salida (si vacío, guarda junto al WAV1)")
    return p.parse_args()

# ---------- Main ----------

def main():
    args = parse_args()

    # Carga(s)
    x1, fs1 = read_wav_mono(args.wav1)
    fs = fs1
    if args.mode == "SSB":
        print(f"[SSB] WAV1: {args.wav1}  Fs={fs1}  side={args.side}  {'FC' if args.carrier>0 else 'SC'}")
        s = ssb_modulate(x1, fs, args.fc, side=args.side, fc_error=args.freqerr,
                         phase_error_deg=args.phase, carrier_full_scale=args.carrier)
        out_path = args.save if args.save else os.path.join(os.path.dirname(args.wav1), "ssb_out.wav")
        sf.write(out_path, s, fs)
        print(f"[SSB] Guardado: {out_path}")

        if args.tx:
            if args.handshake:
                print("[TX] Beeps de arranque…")
                play_triple_beep(fs)
            print("[TX] Transmitiendo SSB…")
            play_audio(s, fs, gain_db=args.gaindb, stereo=True)
            if args.handshake:
                print("[TX] Beep final…")
                play_beep_final(fs)
            print("[TX] OK.")

    else:  # ISB
        if not args.wav2:
            raise ValueError("En modo ISB debes dar --wav2")
        x2, fs2 = read_wav_mono(args.wav2)
        # Alinear Fs si difieren (re-sample simple al mayor)
        if fs1 != fs2:
            fs = max(fs1, fs2)
            if fs1 != fs:
                x1, _ = linear_resample(x1, fs1, fs)
            if fs2 != fs:
                x2, _ = linear_resample(x2, fs2, fs)
        L = min(len(x1), len(x2))
        x1 = x1[:L]; x2 = x2[:L]
        print(f"[ISB] WAV1(USB): {args.wav1}  WAV2(LSB): {args.wav2}  Fs={fs}  {'FC' if args.carrier>0 else 'SC'}")

        s_usb = ssb_modulate(x1, fs, args.fc, side="USB", fc_error=args.freqerr,
                             phase_error_deg=args.phase, carrier_full_scale=0.0)
        s_lsb = ssb_modulate(x2, fs, args.fc, side="LSB", fc_error=args.freqerr,
                             phase_error_deg=args.phase, carrier_full_scale=0.0)
        s = np.clip(s_usb + s_lsb + (args.carrier * np.cos(2*np.pi*args.fc*np.arange(L)/fs + np.deg2rad(args.phase))), -0.95, 0.95).astype(np.float32)

        out_path = args.save if args.save else os.path.join(os.path.dirname(args.wav1), "isb_out.wav")
        sf.write(out_path, s, fs)
        print(f"[ISB] Guardado: {out_path}")

        if args.tx:
            if args.handshake:
                print("[TX] Beeps de arranque…")
                play_triple_beep(fs)
            print("[TX] Transmitiendo ISB…")
            play_audio(s, fs, gain_db=args.gaindb, stereo=True)
            if args.handshake:
                print("[TX] Beep final…")
                play_beep_final(fs)
            print("[TX] OK.")

if __name__ == "__main__":
    main()
