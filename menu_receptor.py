#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===============================================================
# MENU RECEPTOR – Selección y configuración
# ---------------------------------------------------------------
# Flujo pedido:
#  1) Primero se elige el receptor: SSB o Digital.
#  2) Para SSB: elegir entre DEFAULT o ingresar parámetros.
#  3) Para Digital: ejecutar (DEFAULT) o ingresar parámetros (si el script los soporta).
# ===============================================================

import sys
import os
import subprocess
import importlib.util
from pathlib import Path

# ----------------- Rutas/estado -----------------
BASE_DIR   = Path(__file__).resolve().parent
PYTHON_EXE = sys.executable

RX_SSB     = BASE_DIR / "ReceptorSSB.py"
RX_DIGITAL = BASE_DIR / "RxDigital.py"

# ----------------- Utilidades -----------------
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
    s = input(f"{prompt}{f' (def={default})' if default is not None else ''}: ").strip()
    if s == "" and default is not None:
        return default
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

def ask_float(prompt, min_val=None, max_val=None, default=None):
    while True:
        s = input(prompt + (f" (def={default})" if default is not None else "") + ": ").strip()
        if s == "" and default is not None:
            try:
                v = float(default)
            except Exception:
                print("  * Defecto inválido."); continue
        else:
            try:
                v = float(s)
            except Exception:
                print("  * Ingresa un número válido."); continue
        if (min_val is not None) and (v < min_val):
            print(f"  * Debe ser >= {min_val}."); continue
        if (max_val is not None) and (v > max_val):
            print(f"  * Debe ser <= {max_val}."); continue
        return v

def ask_int(prompt, min_val=None, max_val=None, default=None, allow_empty=False):
    while True:
        s = input(prompt + (f" (def={default})" if default is not None else "") + ": ").strip()
        if s == "" and default is not None:
            v = int(default)
        elif s == "" and allow_empty:
            return None
        else:
            try:
                v = int(s)
            except Exception:
                print("  * Ingresa un entero válido."); continue
        if (min_val is not None) and (v < min_val):
            print(f"  * Debe ser >= {min_val}."); continue
        if (max_val is not None) and (v > max_val):
            print(f"  * Debe ser <= {max_val}."); continue
        return v

# ----------------- Carga dinámica -----------------
def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo cargar {path.name}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ----------------- Menú SSB -----------------
def menu_ssb():
    while True:
        clear_screen()
        print("========== RECEPTOR SSB ==========")
        print("1) Ejecutar con valores DEFAULT")
        print("2) Ejecutar ingresando parámetros")
        print("b) Volver")
        print("q) Salir")
        op = ask_str("Opción", default="1").lower()

        if op == "1":
            if not RX_SSB.exists():
                print(f"[ERROR] No se encontró: {RX_SSB}"); pause(); continue
            print(f"[INFO] Ejecutando (DEFAULT): {RX_SSB.name}")
            subprocess.run([PYTHON_EXE, str(RX_SSB)], check=False)
            pause()
        elif op == "2":
            if not RX_SSB.exists():
                print(f"[ERROR] No se encontró: {RX_SSB}"); pause(); continue
            clear_screen()
            print("=== Parámetros SSB (según tu receptor actual) ===\n")
            # SOLO parámetros soportados por el script actual (no tocamos handshake ni lógica interna)
            force_fs  = ask_int("Fs forzada [Hz] (vacío=default del dispositivo)", default=48000, allow_empty=True)
            inp_idx   = ask_int("Índice de dispositivo de entrada (vacío=default)", allow_empty=True)
            fc_hz     = ask_float("Portadora SSB [Hz]", default=12000.0, min_val=300.0, max_val=25000.0)
            bp_bw_hz  = ask_float("Ancho de banda del BP [Hz]", default=4000.0, min_val=500.0, max_val=8000.0)
            bb_cut    = ask_float("Corte LPF baseband [Hz]", default=4500.0, min_val=500.0, max_val=10000.0)
            # Campos alineados con TX (informativos / para matching)
            side     = ask_str("Banda lateral TX (USB/LSB) [informativo]", default="USB").upper()
            while side not in ("USB","LSB"):
                print("  * Debe ser USB o LSB."); side = ask_str("Banda lateral TX (USB/LSB)", default="USB").upper()
            ssb_type = ask_str("Tipo TX: 1=SSB-SC, 2=SSB-FC, 3=ISB [informativo]", default="1")
            # Errores a aplicar en el LO del receptor (sí afectan demod)
            demod_phase = ask_float("Error de fase LO Rx [grados] (0–180)", default=0.0, min_val=0.0, max_val=180.0)
            demod_ferr  = ask_float("Error relativo de frecuencia LO Rx (ej. 0.02 = +2%)", default=0.0, min_val=-0.25, max_val=0.25)
            try:
                rx = _load_module(RX_SSB)
                # Asignaciones respetando variables existentes
                try: rx.FORCE_FS = int(force_fs) if force_fs is not None else 0
                except Exception: pass
                try: rx.INPUT_INDEX = inp_idx if inp_idx is not None else None
                except Exception: pass
                try: rx.SSB_FC_HZ = float(fc_hz)
                except Exception: pass
                try: rx.BP_BW_HZ = float(bp_bw_hz)
                except Exception: pass
                try: rx.BASEBAND_CUT = float(bb_cut)
                except Exception: pass
                # Ajustes de demodulación (sí afectan)
                try: rx.DEMOD_PHASE_DEG = float(demod_phase)
                except Exception: pass
                try: rx.DEMOD_FC_REL_ERR = float(demod_ferr)
                except Exception: pass                # Guard solo si existe en tu script                # PLAY_OUTPUT: algunas versiones usan 1/0
                print("[INFO] Ejecutando receptor SSB con parámetros...")
                rx.run_forever()  # tu script sale tras la primera sesión (tiene break)
                print("[OK] Receptor terminó.")
            except KeyboardInterrupt:
                print("\n[RX] Detenido por usuario.")
            except Exception as e:
                print(f"[ERROR] Al ejecutar receptor SSB: {e}")
            pause()
        elif op == "b":
            return
        elif op in ("q","quit","salir"):
            sys.exit(0)
        else:
            print("Opción no válida."); pause()

# ----------------- Menú Digital -----------------
def menu_digital():
    while True:
        clear_screen()
        print("======== RECEPTOR DIGITAL =========")
        print("1) Ejecutar (DEFAULT) RxDigital.py")
        print("2) Ejecutar ingresando parámetros (si aplica)")
        print("b) Volver")
        print("q) Salir")
        op = ask_str("Opción", default="1").lower()

        if op == "1":
            if not RX_DIGITAL.exists():
                print(f"[ERROR] No se encontró {RX_DIGITAL}"); pause(); continue
            print(f"[INFO] Ejecutando (DEFAULT): {RX_DIGITAL.name}")
            subprocess.run([PYTHON_EXE, str(RX_DIGITAL)], check=False)
            pause()
        elif op == "2":
            if not RX_DIGITAL.exists():
                print(f"[ERROR] No se encontró {RX_DIGITAL}"); pause(); continue
            print("[INFO] (Placeholder) Ejecutando RxDigital.py...")
            subprocess.run([PYTHON_EXE, str(RX_DIGITAL)], check=False)
            pause()
        elif op == "b":
            return
        elif op in ("q","quit","salir"):
            sys.exit(0)
        else:
            print("Opción no válida."); pause()

# ----------------- Main: seleccionar receptor primero -----------------
def main():
    while True:
        clear_screen()
        print("========================================")
        print("         MENU RECEPTOR (Selector)        ")
        print("========================================")
        print("Primero elige cuál receptor usar:")
        print("1) Receptor SSB")
        print("2) Receptor Digital (RxDigital.py)")
        print("q) Salir")

        op = ask_str("Opción", default="1").lower()
        if op == "1":
            menu_ssb()
        elif op == "2":
            menu_digital()
        elif op in ("q","quit","salir"):
            print("Saliendo...")
            break
        else:
            print("Opción no válida."); pause()

if __name__ == "__main__":
    main()
