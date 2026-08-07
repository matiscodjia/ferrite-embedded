#!/usr/bin/env python3
"""Mesure le débit série réel MCU<->hôte via l'echo de uart_bench.rs.

Tourne directement sur le Pi (accès direct à /dev/ttyACM0) — pas besoin du
pont socat pour cette étape de calibration, celui-ci ne sert qu'une fois le
baud choisi, pour relayer vers macOS.

Le firmware doit être flashé avec le même baud que celui passé ici (const
BAUD dans uart_bench.rs) avant de lancer ce script.

Usage: python3 bench_uart.py <baud>
"""
import sys
import time
import os

import serial

PORT = "/dev/ttyACM0"
BLOCK_SIZE = 4096
N_BLOCKS = 20


def bench(baud: int) -> tuple[bool, float, float]:
    ser = serial.Serial(PORT, baudrate=baud, timeout=5)
    time.sleep(0.1)
    ser.reset_input_buffer()

    data = os.urandom(BLOCK_SIZE)
    ok = True
    t0 = time.perf_counter()
    for _ in range(N_BLOCKS):
        ser.write(data)
        echoed = ser.read(BLOCK_SIZE)
        if echoed != data:
            ok = False
    elapsed = time.perf_counter() - t0
    ser.close()

    total_bytes = BLOCK_SIZE * N_BLOCKS * 2  # aller + retour (echo)
    throughput_mbps = (total_bytes * 8) / elapsed / 1e6
    return ok, throughput_mbps, elapsed


if __name__ == "__main__":
    baud = int(sys.argv[1]) if len(sys.argv) > 1 else 921_600
    ok, mbps, elapsed = bench(baud)
    status = "OK" if ok else "CORROMPU (baisser le baud)"
    print(
        f"baud={baud:>9} | {status:>26} | {mbps:6.2f} Mbps | "
        f"{elapsed:.3f}s pour {N_BLOCKS} blocs de {BLOCK_SIZE}o aller-retour"
    )
