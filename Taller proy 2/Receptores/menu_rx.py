#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import importlib.util
import subprocess
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent
PYTHON_EXE = sys.executable

RX_SSB     = BASE_DIR / "ReceptorSSB.py"
RX_DIGITAL = BASE_DIR / "RxDigital.py"

# ------------------------- Utilidades -------------------------

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
        return str(default)
    return s

def ask_float(prompt, min_val=None, max_val=None, default=None):
    while True:
        s = input(prompt + (f" (def={default})" if default is not None else "") + ": ").strip()
        if s == "" and default is not None:
            v = float(default)
        else:
            try:
                v = float(s)
            except Exception:
                print("  * Ingresa un número válido.")
                continue
        if (min_val is not None) and (v < min_val):
            print(f"  * Debe ser >= {min_val}.")
            continue
        if (max_val is not None) and (v > max_val):
            print(f"  * Debe ser <= {max_val}.")
            continue
        return v

def _load_module(path: Path):
    """Carga dinámica de un módulo .py dado su ruta."""
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo cargar {path.name}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _set_attr_safely(obj, name, value):
    """Intenta hacer setattr, ignorando errores si el atributo no existe."""
    try:
        setattr(obj, name, value)
    except Exception:
        pass


# --------------------- Lanzadores analógicos ---------------------

def _run_ssb_base(fc_hz: float, expect_isb: bool, label: str):
    if not RX_SSB.exists():
        print(f"[ERROR] No se encontró: {RX_SSB}")
        pause()
        return
    rx = _load_module(RX_SSB)
    _set_attr_safely(rx, "SSB_FC_HZ", float(fc_hz))
    _set_attr_safely(rx, "EXPECT_ISB", bool(expect_isb))
    _set_attr_safely(rx, "RX_EXPECT_FC", float(fc_hz))
    print(f"[INFO] Ejecutando receptor {label} (f_c={fc_hz:.1f} Hz) ...")
    if hasattr(rx, "run_forever"):
        rx.run_forever()
    elif hasattr(rx, "receptor_main"):
        rx.receptor_main()
    else:
        raise RuntimeError("El receptor no expone run_forever() ni receptor_main().")
    print("[OK] Receptor terminó.")

def run_ssb_fc(fc_hz: float):
    _run_ssb_base(fc_hz, expect_isb=False, label="SSB-FC")

def run_ssb_sc(fc_hz: float):
    _run_ssb_base(fc_hz, expect_isb=False, label="SSB-SC")

def run_isb(fc_hz: float):
    _run_ssb_base(fc_hz, expect_isb=True, label="ISB")


# --------------------------- Menús ----------------------------

def menu_analog():
    while True:
        clear_screen()
        print("=========== RECEPTOR ANALÓGICO ===========")
        print("1) SSB-FC (con portadora)")
        print("2) SSB-SC (sin portadora)")
        print("3) ISB (dos bandas laterales independientes)")
        print("b) Volver")
        op = ask_str("Opción", default="1").lower()
        if op == "b":
            return

        if op not in ("1", "2", "3"):
            print("Opción no válida.")
            pause()
            continue

        fc = ask_float("Frecuencia portadora f_c [Hz]", min_val=1000.0, max_val=22000.0, default=12000.0)
        try:
            if op == "1":
                run_ssb_fc(fc)
            elif op == "2":
                run_ssb_sc(fc)
            else:
                run_isb(fc)
        except KeyboardInterrupt:
            print("\n[RX] Detenido por usuario.")
            pause()
        except Exception as e:
            print(f"[ERROR] {e}")
            pause()

def menu_digital():
    while True:
        clear_screen()
        print("=========== RECEPTOR DIGITAL (BPSK) ===========")
        print("1) Ejecutar receptor digital BPSK")
        print("b) Volver")
        op = ask_str("Opción", default="1").lower()
        if op == "b":
            return
        if op == "1":
            if not RX_DIGITAL.exists():
                print(f"[ERROR] No se encontró {RX_DIGITAL}")
                pause()
                continue
            # Preguntar tipo de archivo esperado para reconstruirlo con esa extensión
            tipo = ask_str("Tipo de archivo recibido (ej: jpg, png, wav, pdf, bin)", default="bin")
            tipo = (tipo or "bin").strip().lower()
            if not tipo:
                tipo = "bin"
            # Pasamos la extensión como primer argumento a RxDigital.py
            print(f"[INFO] Ejecutando: {RX_DIGITAL.name} para archivo .{tipo}")
            try:
                subprocess.run([PYTHON_EXE, str(RX_DIGITAL), tipo], check=False)
            except Exception as e:
                print(f"[ERROR] Falló la ejecución de RxDigital.py: {e}")
            pause()
        else:
            print("Opción no válida.")
            pause()

def main():
    while True:
        clear_screen()
        print("========================================")
        print("              MENU RECEPTOR             ")
        print("========================================")
        print("1) Receptor analógico")
        print("2) Receptor digital")
        print("q) Salir")
        op = ask_str("Opción", default="1").lower()
        if op == "1":
            menu_analog()
        elif op == "2":
            menu_digital()
        elif op in ("q", "quit", "salir"):
            print("Saliendo...")
            break
        else:
            print("Opción no válida.")
            pause()

if __name__ == "__main__":
    main()
