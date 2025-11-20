#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===============================================================
# MENU TX – SSB / ISB / Digital (mínimo de preguntas)
# ---------------------------------------------------------------
# - SSB: pides WAV1, SC/FC y USB/LSB y listo (el resto default).
# - ISB: pides WAV1, WAV2 y si incluir FC (el resto default).
# - Digital: ejecuta TxDigital.py (si existe).
# - Invoca OpenWAV.py con argumentos CLI (sin stdin interactivo).
# ===============================================================

import sys
import os
import subprocess
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent
PYTHON_EXE = sys.executable

OPENWAV    = BASE_DIR / "OpenWAV.py"
TXDIGITAL  = BASE_DIR / "TxDigital.py"

# -------------------- Utils --------------------

def clear_screen():
    try:
        os.system("cls" if os.name == "nt" else "clear")
    except Exception:
        pass

def pause(msg="\\nPresiona Enter para continuar..."):
    try:
        input(msg)
    except EOFError:
        pass

def ask_str(prompt, default=None):
    s = input(f"{prompt}{f' (def={default})' if default is not None else ''}: ").strip()
    if s == "" and default is not None:
        return str(default)
    return s

def ask_yesno(prompt, default=None):
    suf = " [s/n]"
    if isinstance(default, bool):
        suf = " [S/n]" if default else " [s/N]"
    while True:
        s = input(prompt + suf + ": ").strip().lower()
        if s == "" and default is not None:
            return bool(default)
        if s in ("s","si","sí","y","yes"):
            return True
        if s in ("n","no"):
            return False
        print("  * Responde con s/n.")

def ask_float(prompt, min_val=None, max_val=None, default=None, percent=False):
    while True:
        s = input(prompt + (f" (def={default})" if default is not None else "") + ": ").strip()
        if s == "" and default is not None:
            v = float(default)
        else:
            try:
                if percent and s.endswith("%"):
                    v = float(s[:-1]) / 100.0
                else:
                    v = float(s)
                if percent and abs(v) > 1.0:
                    v = v / 100.0
            except Exception:
                print("  * Ingresa un número válido."); continue
        if (min_val is not None) and (v < min_val):
            print(f"  * Debe ser >= {min_val}."); continue
        if (max_val is not None) and (v > max_val):
            print(f"  * Debe ser <= {max_val}."); continue
        return v

# -------------------- Launchers --------------------

def run_tx_ssb():
    clear_screen()
    print("========== TRANSMISOR SSB ==========")

    if not OPENWAV.exists():
        print(f"[ERROR] No se encontró: {OPENWAV}")
        pause(); return

    wav1 = ask_str("Ruta WAV 1 (incluye .wav)")
    if not wav1.lower().endswith(".wav"):
        print("[ERROR] Debes ingresar la ruta con extensión .wav"); pause(); return

    print("Tipo de SSB:")
    print("  1) SSB-SC (sin portadora)")
    print("  2) SSB-FC (con portadora)")
    op_type = ask_str("Elige 1 o 2", default="1")
    ssb_fc = (op_type == "2")

    side = ask_str("Banda lateral (USB/LSB)", default="USB").upper()
    if side not in ("USB","LSB"):
        print("[ERROR] Debe ser USB o LSB."); pause(); return

    # Defaults: fc=12000, phase=0, freqerr=0, carrier_level=0.1 si FC else 0.0
    args = [PYTHON_EXE, str(OPENWAV),
            "--wav1", wav1,
            "--mode", "SSB",
            "--side", side,
            "--fc", "12000",
            "--phase", "0",
            "--freqerr", "0",
            "--tx", "1",
            "--handshake", "1",
            "--gaindb", "0"
           ]
    if ssb_fc:
        args += ["--carrier", "0.1"]
    else:
        args += ["--carrier", "0.0"]

    print(f"[INFO] Ejecutando: OpenWAV.py (SSB {('FC' if ssb_fc else 'SC')} / {side})")
    subprocess.run(args, check=False)
    print("[OK] TX SSB terminó.")
    pause()

def run_tx_isb():
    clear_screen()
    print("========== TRANSMISOR ISB (USB + LSB) ==========")

    if not OPENWAV.exists():
        print(f"[ERROR] No se encontró: {OPENWAV}")
        pause(); return

    wav1 = ask_str("Ruta WAV (USB)  (incluye .wav)")
    if not wav1.lower().endswith(".wav"):
        print("[ERROR] Debes ingresar la ruta con extensión .wav"); pause(); return

    wav2 = ask_str("Ruta WAV (LSB)  (incluye .wav)")
    if not wav2.lower().endswith(".wav"):
        print("[ERROR] Debes ingresar la ruta con extensión .wav"); pause(); return

    inc_fc = ask_yesno("¿Incluir portadora (FC)?", default=False)

    # Defaults: fc=12000, phase=0, freqerr=0, carrier_level=0.1 si FC else 0.0
    args = [PYTHON_EXE, str(OPENWAV),
            "--wav1", wav1,
            "--wav2", wav2,
            "--mode", "ISB",
            "--fc", "12000",
            "--phase", "0",
            "--freqerr", "0",
            "--tx", "1",
            "--handshake", "1",
            "--gaindb", "0"
           ]
    if inc_fc:
        args += ["--carrier", "0.1"]
    else:
        args += ["--carrier", "0.0"]

    print("[INFO] Ejecutando: OpenWAV.py (ISB)")
    subprocess.run(args, check=False)
    print("[OK] TX ISB terminó.")
    pause()

def run_txdigital():
    clear_screen()
    if not TXDIGITAL.exists():
        print(f"[ERROR] No se encontró: {TXDIGITAL}")
        pause(); return
    print(f"[INFO] Ejecutando: {TXDIGITAL.name}")
    subprocess.run([PYTHON_EXE, str(TXDIGITAL)], check=False)
    print("[OK] TxDigital terminó.")
    pause()

# -------------------- Main --------------------

def main():
    while True:
        clear_screen()
        print("========================================")
        print("              MENU TRANSMISOR           ")
        print("========================================")
        print("1) Transmisor SSB")
        print("2) Transmisor ISB")
        print("3) Modulación digital pasobanda")
        print("q) Salir")

        op = ask_str("Opción", default="1").lower()
        if op == "1":
            run_tx_ssb()
        elif op == "2":
            run_tx_isb()
        elif op == "3":
            run_txdigital()
        elif op in ("q","quit","salir"):
            print("Saliendo..."); break
        else:
            print("Opción no válida."); pause()

if __name__ == "__main__":
    main()
