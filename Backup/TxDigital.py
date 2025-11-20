#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===================================================================
#           TRANSMISOR DIGITAL PASOBANDA (QPSK)
# -------------------------------------------------------------------
# DESCRIPCIÓN
#
#
# USO
#   python3 TxDigital.py --text "hola mundo" --out TX_QPSK.wav
#   python3 TxDigital.py --fs 48000 --fc 8000 --rs 2000 --osr 12 --out tx.wav
#
# DEPENDENCIAS
# - numpy, soundfile
#
# ===================================================================

import argparse
import json
import numpy as np
import soundfile as sf

# ============================ PARÁMETROS ============================

FS_DEFAULT         = 48_000        # tasa de muestreo de salida
FC_DEFAULT         = 12_000        # portadora audio
RS_DEFAULT         = 1_500         # símbolos/s (QPSK)
OSR_DEFAULT        = 12            # muestras por símbolo
BETA_DEFAULT       = 0.35          # rolloff
SPAN_SYM_DEFAULT   = 8             # longitud del RRC en símbolos [usar valores pares]

# BEEPS
BEEP_FREQ          = 1000.0        # Hz
BEEP_MS            = 200.0         # ms
BEEP_GAP_S         = 0.80          # s entre beeps
OUT_GAIN           = 0.80          # [-1,1]

# ============================ TOOLS ============================

def text_to_bits(s: str, encoding: str = "utf-8") -> np.ndarray: #CONVERSIÓN STRING A BITS (MSB primero)
    b = s.encode(encoding)
    bits = np.unpackbits(np.frombuffer(b, dtype=np.uint8))
    return bits.astype(np.uint8)


def bytes_to_bits(data: bytes) -> np.ndarray: #BYTES A BITS (MSB primero)
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(np.uint8)


def barker13_bits() -> np.ndarray: #SECUENCIA BARKER-13 EN BITS
    seq = [1,1,1,1,1,0,0,1,1,0,1,0,1]
    return np.array(seq, dtype=np.uint8)


def rrc_taps(beta: float = 0.35, sps: int = 8, span: int = 8) -> np.ndarray: #FILTRO RRC
    N = span * sps
    t = (np.arange(-N//2, N//2 + 1) / sps).astype(np.float64)
    taps = np.zeros_like(t, dtype=np.float64)

    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            taps[i] = 1.0 - beta + (4*beta/np.pi)
        elif beta > 0 and abs(abs(4*beta*ti) - 1.0) < 1e-12:
            # singularidad en t = ±1/(4β)
            a = (1 + 2/np.pi)*np.sin(np.pi/(4*beta))
            b = (1 - 2/np.pi)*np.cos(np.pi/(4*beta))
            taps[i] = (a + b)
        else:
            num = np.sin(np.pi*ti*(1 - beta)) + 4*beta*ti*np.cos(np.pi*ti*(1 + beta))
            den = np.pi*ti*(1 - (4*beta*ti)**2)
            taps[i] = num/den

    # Normalización de energía
    taps /= np.sqrt(np.sum(taps**2) + 1e-18)
    return taps


def upsample_and_filter(symbols: np.ndarray, sps: int, taps: np.ndarray | None) -> np.ndarray: #UPSAMPLE + FILTRADO
    up = np.zeros(len(symbols) * sps, dtype=np.float64)
    up[::sps] = symbols.astype(np.float64)
    if taps is None:
        return up
    return np.convolve(up, taps, mode='same')


def normalize_audio(x: np.ndarray, peak: float = OUT_GAIN) -> np.ndarray: #NORMALIZACIÓN DE NIVEL
    m = float(np.max(np.abs(x)) + 1e-12)
    return (x / m) * peak

def tone(freq: float, dur_s: float, fs: int) -> np.ndarray:
    t = np.arange(int(dur_s * fs)) / fs
    return np.sin(2 * np.pi * freq * t)

def beeps_sync(fs: int) -> np.ndarray: #BEEPS DE SINCRONÍA
    beep = tone(BEEP_FREQ, BEEP_MS / 1000.0, fs)
    gap  = np.zeros(int(BEEP_GAP_S * fs))
    return np.concatenate([beep, gap, beep, gap, beep])

# =========================================== MAPEO QPSK ===========================================

def map_qpsk(bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pad = (-len(bits)) % 2
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    b0 = bits[0::2]
    b1 = bits[1::2]
    I = 2 * b0 - 1.0
    Q = 2 * b1 - 1.0
    return I.astype(np.float64), Q.astype(np.float64)

# =========================================== MODULACIÓN PASOBANDA ===========================================

def upconvert_iq(i: np.ndarray, q: np.ndarray, fs: int, fc: float) -> np.ndarray:
    n = np.arange(len(i))
    c = np.cos(2 * np.pi * fc * n / fs)
    s = np.sin(2 * np.pi * fc * n / fs)
    return i * c - q * s

# =========================================== FRAMING ===========================================

def make_frame(payload_bits: np.ndarray, add_len: bool = True) -> np.ndarray:
    barker = barker13_bits()  # 13 bits
    if add_len:
        L_bytes = len(payload_bits) // 8
        L0 = (L_bytes) & 0xFF
        L1 = (L_bytes >> 8) & 0xFF
        L_bits = np.unpackbits(np.array([L0, L1], dtype=np.uint8))
        hdr = np.concatenate([barker, L_bits.astype(np.uint8)])
    else:
        hdr = barker
    return np.concatenate([hdr, payload_bits])

# =========================================== PIPELINE TX (QPSK) ===========================================

def tx_qpsk_waveform(bits: np.ndarray,
                     fs: int,
                     rs: int,
                     fc: float,
                     osr: int,
                     beta: float,
                     span_sym: int,
                     use_rrc: bool = True) -> tuple[np.ndarray, dict]:
    # 1) Trama
    frame_bits = make_frame(bits, add_len=True)

    # 2) QPSK Gray
    I_sym, Q_sym = map_qpsk(frame_bits)

    # 3) Shaping
    sps  = osr
    taps = rrc_taps(beta=beta, sps=sps, span=span_sym) if use_rrc else None
    I_bb = upsample_and_filter(I_sym, sps, taps)
    Q_bb = upsample_and_filter(Q_sym, sps, taps)

    # 4) Pasobanda
    y = upconvert_iq(I_bb, Q_bb, fs, fc)

    # 5) Beeps
    prefix = beeps_sync(fs)
    y = np.concatenate([prefix, y])

    # 6) Normalización
    y = normalize_audio(y, peak=OUT_GAIN)

    meta = {
        "fs": fs, "fc": fc, "rs": rs, "osr": osr,
        "beta": beta, "span_sym": span_sym, "mod": "qpsk",
        "len_bits_payload": int(len(bits)),
        "len_bits_frame": int(len(frame_bits)),
        "beeps_hz": BEEP_FREQ, "beeps_ms": BEEP_MS, "beeps_gap_s": BEEP_GAP_S,
        "rrc": bool(use_rrc)
    }
    return y, meta

# =========================================== MAIN ===========================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transmisor digital pasobanda — QPSK (Fase 1, sin FEC)")
    p.add_argument("--text", type=str, default="hola mundo",
                   help="Texto a transmitir (alternativa a --bin)")
    p.add_argument("--bin", type=str, default=None,
                   help="Archivo binario a transmitir (se prioriza sobre --text)")
    p.add_argument("--fs", type=int, default=FS_DEFAULT, help="Frecuencia de muestreo [Hz]")
    p.add_argument("--fc", type=float, default=FC_DEFAULT, help="Portadora [Hz]")
    p.add_argument("--rs", type=int, default=RS_DEFAULT, help="Symbol rate [sym/s]")
    p.add_argument("--osr", type=int, default=OSR_DEFAULT, help="Oversampling [samp/sym]")
    p.add_argument("--beta", type=float, default=BETA_DEFAULT, help="Rolloff RRC (0..1)")
    p.add_argument("--span", type=int, default=SPAN_SYM_DEFAULT, help="Longitud RRC en símbolos")
    p.add_argument("--no-rrc", action="store_true", help="Desactivar pulse shaping RRC (pulso rectangular)")
    p.add_argument("--out", type=str, default="TX_QPSK.wav", help="Archivo WAV de salida")
    p.add_argument("--meta", type=str, default="TX_QPSK_meta.json", help="Archivo JSON de metadatos")
    return p.parse_args()

def main():
    args = _parse_args()

    # Fuente de payload: binario prioriza sobre texto
    if args.bin is not None:
        with open(args.bin, "rb") as f:
            payload = f.read()
        bits = bytes_to_bits(payload)
        src_desc = f"bin:{args.bin}"
    else:
        bits = text_to_bits(args.text)
        src_desc = f"text:{args.text[:32]}"

    # Pipeline TX
    y, meta = tx_qpsk_waveform(
        bits=bits,
        fs=args.fs,
        rs=args.rs,
        fc=args.fc,
        osr=args.osr,
        beta=args.beta,
        span_sym=args.span,
        use_rrc=(not args.no_rrc)
    )

    # Salidas
    sf.write(args.out, y.astype(np.float32), args.fs)
    with open(args.meta, "w") as f:
        json.dump(meta, f, indent=2)

    # Log mínimo a consola
    print("[OK] WAV:", args.out)
    print("[OK] META:", args.meta)
    print(f"     src={src_desc}, fs={args.fs} Hz, fc={args.fc} Hz, rs={args.rs} sym/s, osr={args.osr}, RRC={not args.no_rrc}")

if __name__ == "__main__":
    main()

