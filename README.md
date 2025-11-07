# Proyecto2TallerComu — Modulación/Demodulación SSB e ISB

## EL-5622 – Comunicaciones Eléctricas 2 

[![Estado](https://img.shields.io/badge/estado-en_desarrollo-blue.svg)]()
[![Plataforma](https://img.shields.io/badge/target-Ubuntu%2024.04-green.svg)]()
[![Build](https://img.shields.io/badge/build-Python%203.10%2B-pink.svg)]()
[![Lenguajes](https://img.shields.io/badge/lenguajes-Python-orange.svg)]()
[![Stack](https://img.shields.io/badge/stack-NumPy%20%7C%20SciPy%20%7C%20SoundDevice%20%7C%20PyAudio%20%7C%20SoundFile-lightgrey.svg)]()

---

## 1. Introducción

<p align="justify">
Este documento describe el desarrollo de un sistema de <strong>modulación/demodulación de Banda Lateral Única (SSB)</strong> y <strong>Banda Lateral Independiente (ISB)</strong> mediante el enfoque de la <strong>Transformada de Hilbert</strong>. El flujo soporta entrada/salida <strong>WAV</strong>, generación de <em>DSB/SSB/ISB</em>, transmisión por parlante (TX) y recepción/demodulación por micrófono (RX). Se incluyen controles de <strong>error de fase</strong> y <strong>error de frecuencia</strong> para evaluar la sensibilidad del demodulador síncrono.
</p>

---

## 2. Integrantes

- Christopher Quirós
- Alonso Román

---

## 3. Alcance y requisitos del enunciado

**SSB/ISB (audio):**
- Entrada **WAV** (mono/estéreo → mono) y salida **WAV**.
- Portadora: \( f_c \le 25~\text{kHz} \). El sistema remuestrea a **48 kHz** cuando es necesario.
- Modos: **SSB-SC** (suprime portadora), **SSB-FC** (añade portadora), **ISB** (dos WAV, uno por banda).
- Errores: fase \(0–180^\circ\) y frecuencia relativa \(\pm 25\%\).
- Evidencias: tiempo/espectro de mensaje, DSB, SSB/ISB y mensaje recuperado; reproducción del audio recuperado.

**Digital pasobanda (interfaz):**
- Selección de archivo digital, con/sin FEC y tipo de modulación (definición e implementación en iteraciones siguientes; interfaz ya incluida).

---

## 4. Arquitectura de señal

- **Tx (SSB/ISB).** Se forma la señal analítica \(x_a(t)=x(t)+j\,\mathcal{H}\{x(t)\}\). La banda (USB/LSB) se obtiene por combinación en cuadratura. En **SSB-FC** se añade portadora con nivel configurable; en **ISB** se suman dos mensajes independientes (USB y LSB) al mismo \(f_c\).
- **Rx (síncrono).** Producto con oscilador local y **LPF** para recuperar \(x(t)\). Se permite desajuste \(\Delta f\) y \(\Delta \phi\). **BPF** previo opcional alrededor de \(f_c\) antes de la demodulación para robustecer el canal acústico.
- **Handshake acústico.** 3 beeps de arranque y beep final para coordinar la captura.

---

## 5. Menú en terminal

El proyecto incluye un **menú interactivo** que solicita **dato por dato**:

**Flujo SSB/ISB (Tx):**
1. Ruta de **WAV 1** (y **WAV 2** si ISB).  
2. **\(f_c\)** (0–25 kHz).  
3. **USB/LSB** (si SSB simple).  
4. **SSB-SC** o **SSB-FC** (nivel de portadora 0–1).  
5. **Error de fase** [°] y **error relativo de frecuencia** (acepta “2” o “2%”).  

---

## 6. Procedimiento de modulación (Tx)

1. **Lectura WAV** → conversión a **mono** y remuestreo a **48 kHz**.  
2. **Hilbert** → señal analítica.  
3. **SSB**: combinación en cuadratura → **USB** o **LSB**.  
4. **FC opcional**: suma de la portadora.  
5. **ISB**: **WAV 1 → USB** y **WAV 2 → LSB** (misma \(f_s\), mismo \(f_c\)); normalización para evitar clipping.  
6. **Salida**: guardar WAV y/o transmitir por parlante con handshake.

---

## 7. Procedimiento de demodulación (Rx)

1. **Captura** por micrófono.  
2. **BPF opcional** alrededor de \(f_c\).  
3. **Demodulación sincro**: producto con LO \(\cos(2\pi(f_c+\Delta f)t+\Delta \phi)\).  
4. **LPF** para recuperar \(x(t)\).  
5. **Ajustes**: barrer \(\Delta f\) y \(\Delta \phi\) para cuantificar degradación.  
6. **Salida**: WAV recuperado y reproducción para evaluación subjetiva.

---


## 8. Instalación rápida

**Dependencias del sistema (Ubuntu 24.04):**
```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev build-essential libsndfile1 portaudio19-dev
```


## 9. Comandos reproducir WAV y ejecutar TX/RX

**Linux:**
```bash
aplay Entrada.wav
aplay Demodulado.wav

#Transmisor
python3 menu.py

#Receptor
python3 ReceptorSSB.py
```
