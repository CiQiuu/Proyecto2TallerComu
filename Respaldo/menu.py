#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===============================================================
# MENU CLI – Lanzador puro (sin lógica de modulación interna)
# ---------------------------------------------------------------
# Este menú NO reimplementa nada de OpenWAV.py ni TxDigital.py.
# Solo invoca esos scripts como subprocesos.
#
# Grupos:
#  1) Rutas/constantes        -> BASE_DIR, PYTHON_EXE
#  2) Lanzadores              -> run_openwav_default, run_openwav_interactive, run_txdigital
#  3) Utilidades              -> clear_screen, pause, ask_str, ask_yesno, ask_float
#  4) Menú principal          -> main()
# ===============================================================

import sys
import os
import subprocess
from pathlib import Path

# ===================== GRUPO 1 — Rutas/constantes =====================
BASE_DIR   = Path(__file__).resolve().parent
PYTHON_EXE = sys.executable


# ===================== GRUPO 2 — Lanzadores ===========================
def run_openwav_default():

    script = BASE_DIR / "OpenWAV.py"
    if not script.exists():
        print(f"[ERROR] No se encontró: {script}")
        return
    print(f"[INFO] Ejecutando (DEFAULT): {script.name}")
    try:
        subprocess.run([PYTHON_EXE, str(script)], check=False)
        print("[OK] Tx terminó.")
    except Exception as e:
        print(f"[ERROR] Al ejecutar Tx: {e}")


def run_openwav_interactive():

    script = BASE_DIR / "OpenWAV.py"
    if not script.exists():
        print(f"[ERROR] No se encontró: {script}")
        return

    clear_screen()
    print("=== MODULACIÓN SSB / ISB ===\n")

    # ------------- Captura de parámetros en el menú -------------
    wav1 = ask_str("Ruta de WAV 1")
    isb  = ask_yesno("¿Usar ISB (dos WAV, uno por banda)?", default=False)
    wav2 = ""
    if isb:
        wav2 = ask_str("Ruta de WAV 2 (solo si ISB)")

    fc = ask_float("Frecuencia de portadora [Hz] (0 < f_c ≤ 25000)", min_val=0.1, max_val=25_000.0, default=12_000.0)

    side = ""
    ssb_type = ""
    carrier_level = "0.0"
    if not isb:
        side = ask_str("Banda lateral a enviar (USB/LSB)", default="USB").upper()
        while side not in ("USB", "LSB"):
            print("  * Debe ser USB o LSB.")
            side = ask_str("Banda lateral a enviar (USB/LSB)", default="USB").upper()
        print("Tipo de modulación SSB:")
        print("  1) SSB-SC (sin portadora)")
        print("  2) SSB-FC (con portadora)")
        ssb_type = ask_str("Elige 1 o 2", default="1")
        if ssb_type == "2":
            carrier_level = str(ask_float("Nivel de portadora (0–1, ej. 0.1)", min_val=0.0, max_val=1.0, default=0.1))
    else:
        include_fc = ask_yesno("ISB: ¿incluir portadora (FC)?", default=False)
        if include_fc:
            carrier_level = str(ask_float("Nivel de portadora (0–1, ej. 0.1)", min_val=0.0, max_val=1.0, default=0.1))

    phase_deg       = ask_float("Error de fase [grados] (0–180)", min_val=0.0, max_val=180.0, default=0.0)
    freq_error_rel  = ask_float("Error relativo de frecuencia (±25%) (puedes escribir 2 o 2%)",
                                min_val=-0.25, max_val=0.25, default=0.0, percent=True)

    out_wav = ask_str("Ruta para guardar WAV de salida (enter = no guardar)", default="")
    do_play = ask_yesno("¿Transmitir?", default=False)
    do_hs   = False
    gain_db = 0.0
    if do_play:
        do_hs   = ask_yesno("¿Usar handshake (3 beeps + beep final)?", default=True)
        gain_db = ask_float("Ganancia TX [dB] (ej. 0)", default=0.0)

    # ------------- Construcción de stdin para OpenWAV.py -------------

    answers = []
    answers.append(wav1)                    # Ruta WAV 1
    answers.append("s" if isb else "n")     # ¿ISB?
    if isb:
        answers.append(wav2)                # Ruta WAV 2
    answers.append(str(fc))                 # f_c (Hz)

    if not isb:
        answers.append(side)                # USB/LSB
        answers.append("2" if ssb_type == "2" else "1")  # 1=SC, 2=FC
        if ssb_type == "2":
            answers.append(carrier_level)   # nivel portadora
    else:
        pass

    answers.append(str(phase_deg))          # error fase [°]
    answers.append(f"{freq_error_rel}")     # error frecuencia relativo
    answers.append(out_wav)                 # ruta salida (o vacío)
    answers.append("s" if do_play else "n") # ¿TX?
    if do_play:
        answers.append("s" if do_hs else "n")    # ¿handshake?
        answers.append(str(gain_db))             # ganancia dB

    payload = ("\n".join(answers) + "\n").encode("utf-8")

    print(f"[INFO] Ejecutando (INTERACTIVO vía stdin): {script.name}")
    try:
        subprocess.run([PYTHON_EXE, str(script)], input=payload, check=False)
        print("[OK] Tx terminó.")
    except Exception as e:
        print(f"[ERROR] Al ejecutar Tx: {e}")


def run_txdigital():

    script = BASE_DIR / "TxDigital.py"
    if not script.exists():
        print(f"[ERROR] No se encontró: {script}")
        print("        Crea TxDigital.py o renómbralo correctamente.")
        return
    print(f"[INFO] Ejecutando: {script.name}")
    try:
        subprocess.run([PYTHON_EXE, str(script)], check=False)
        print("[OK] TxDigital.py terminó.")
    except Exception as e:
        print(f"[ERROR] Al ejecutar TxDigital.py: {e}")


# ===================== GRUPO 3 — Utilidades ===========================
def clear_screen():
    try:
        os.system("cls" if os.name == "nt" else "clear")
    except Exception:
        pass

def pause(msg="\nPresiona Enter para continuar..."):
    try:
        input(msg)
    except EOFError:
        pass

def ask_str(prompt, default=None):
    while True:
        s = input(f"{prompt}{f' (def={default})' if default is not None else ''}: ").strip()
        if s == "" and default is not None:
            return default
        if s != "":
            return s
        print("  * Valor no puede ser vacío.")

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
                print("  * Ingresa un número válido.")
                continue
        if (min_val is not None) and (v < min_val):
            print(f"  * Debe ser >= {min_val}."); continue
        if (max_val is not None) and (v > max_val):
            print(f"  * Debe ser <= {max_val}."); continue
        return v


# ===================== GRUPO 4 — Menú principal =======================
def main():
    while True:
        clear_screen()
        print("========================================")
        print("   PROYECTO 2 – MENU INTERACTIVO CLI    ")
        print("========================================")
        print("1) Modulación SSB / ISB")
        print("2) Modulación digital pasobanda")
        print("0) Salir")

        op = ask_str("Elige una opción", default="1")

        if op == "1":
            clear_screen()
            print("=== Opción 1: SSB / ISB ===\n")
            print("1) Usar valores DEFAULT de modulacion")
            print("2) Ingresar valores para modulacion")
            print("0) Volver")
            sub = ask_str("Elige una opción", default="1")
            if sub == "1":
                clear_screen()
                run_openwav_default()
                pause()
            elif sub == "2":
                clear_screen()
                run_openwav_interactive()
                pause()
            else:
                pass  # volver

        elif op == "2":
            clear_screen()
            run_txdigital()
            pause()

        elif op == "0":
            print("Saliendo...")
            break
        else:
            print("Opción no válida."); pause()


if __name__ == "__main__":
    main()

