#!/usr/bin/env python3
"""Analyse automatique des logs RTT de regime_bench / stream_bench.

Les deux firmwares impriment, en plus des lignes lisibles, une ligne
compacte par régime préfixée par "REGIME|" ou "STREAM|" — ce script les
extrait (peu importe le préfixe [INFO]/[WARN] ou le suffixe
"(bin src/bin/x.rs:NN)" que probe-rs ajoute autour) et produit un rapport
Markdown : tableau récapitulatif + verdict de synthèse.

Usage:
    python3 scripts/analyze.py scripts/logs/regime_bench-20260807-120000.log
    python3 scripts/analyze.py scripts/logs/stream_bench-*.log --out report.md
    python3 scripts/analyze.py some.log another.log   # plusieurs logs à la fois

Aucune dépendance hors stdlib — voulu portable sans setup côté carte.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

REGIME_FIELDS = [
    "name", "H", "W", "C", "K", "KH", "KW", "macs", "ram_bytes",
    "status", "cycles_per_iter", "cycles_per_mac", "time_us", "pct_tick",
]
STREAM_FIELDS = [
    "name", "W", "KH", "KW", "cycles_per_row", "us_per_row", "cycles_per_mac",
    "fps_120", "fps_240", "fps_480", "fps_720",
]

RAM_TOTAL_BYTES = 128 * 1024
TICK_BUDGET_US = 10_000.0


def _extract_tagged(line: str, tag: str) -> str | None:
    """Isole la charge utile d'une ligne "...REGIME|a|b|c... (bin file:NN)",
    en tolérant le préfixe defmt/probe-rs et le suffixe "(fichier:ligne)"."""
    idx = line.find(tag)
    if idx == -1:
        return None
    payload = line[idx:]
    paren = payload.rfind(" (")
    if paren != -1:
        payload = payload[:paren]
    return payload.strip()


def _cast(value: str):
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_log(path: Path) -> tuple[list[dict], list[dict]]:
    regimes, streams = [], []
    for raw_line in path.read_text(errors="replace").splitlines():
        if (payload := _extract_tagged(raw_line, "REGIME|")) is not None:
            parts = payload.split("|")[1:]  # drop the "REGIME" tag itself
            if len(parts) != len(REGIME_FIELDS):
                continue
            regimes.append({k: _cast(v) for k, v in zip(REGIME_FIELDS, parts)})
        elif (payload := _extract_tagged(raw_line, "STREAM|")) is not None:
            parts = payload.split("|")[1:]
            if len(parts) != len(STREAM_FIELDS):
                continue
            streams.append({k: _cast(v) for k, v in zip(STREAM_FIELDS, parts)})
    return regimes, streams


def _verdict(pct_tick: float) -> str:
    if pct_tick <= 50.0:
        return "OK"
    if pct_tick <= 100.0:
        return "SERRÉ"
    return "DÉPASSE"


def render_regime_report(regimes: list[dict]) -> str:
    if not regimes:
        return ""
    lines = [
        "## Régimes de convolution (`regime_bench`)",
        "",
        "| Régime | H×W | C | K | Noyau | MACs | RAM | cycles/MAC | temps/iter | % tick 10ms | Verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in regimes:
        if r["status"] == "SKIP":
            ram_pct = 100.0 * r["ram_bytes"] / RAM_TOTAL_BYTES
            lines.append(
                f"| {r['name']} | {r['H']}×{r['W']} | {r['C']} | {r['K']} | "
                f"{r['KH']}×{r['KW']} | {r['macs']:,} | {r['ram_bytes']:,} o ({ram_pct:.0f}%) | "
                f"— | — | — | **SKIP (RAM)** |"
            )
            continue
        ram_pct = 100.0 * r["ram_bytes"] / RAM_TOTAL_BYTES
        verdict = _verdict(r["pct_tick"])
        lines.append(
            f"| {r['name']} | {r['H']}×{r['W']} | {r['C']} | {r['K']} | "
            f"{r['KH']}×{r['KW']} | {r['macs']:,} | {r['ram_bytes']:,} o ({ram_pct:.0f}%) | "
            f"{r['cycles_per_mac']:.2f} | {r['time_us']:.0f} µs | {r['pct_tick']:.1f}% | **{verdict}** |"
        )

    ok = [r for r in regimes if r["status"] == "OK" and _verdict(r["pct_tick"]) == "OK"]
    tight = [r for r in regimes if r["status"] == "OK" and _verdict(r["pct_tick"]) == "SERRÉ"]
    over = [r for r in regimes if r["status"] == "OK" and _verdict(r["pct_tick"]) == "DÉPASSE"]
    skipped = [r for r in regimes if r["status"] == "SKIP"]

    lines += ["", "**Synthèse** :"]
    lines.append(
        f"- {len(ok)} régime(s) confortables (≤50% du tick), "
        f"{len(tight)} serré(s) (50-100%), {len(over)} dépassent le tick à eux seuls, "
        f"{len(skipped)} écartés d'office pour RAM insuffisante."
    )
    if ok:
        best = max(ok, key=lambda r: r["macs"])
        lines.append(
            f"- Régime le plus exigeant qui reste confortable : **{best['name']}** "
            f"({best['H']}×{best['W']}, C={best['C']}, K={best['K']}) — "
            f"{best['pct_tick']:.1f}% du tick, {100.0 * best['ram_bytes'] / RAM_TOTAL_BYTES:.0f}% de la RAM."
        )
    macs = [r["cycles_per_mac"] for r in regimes if r["status"] == "OK"]
    if macs:
        lines.append(
            f"- cycles/MAC mesuré : min {min(macs):.2f}, max {max(macs):.2f} "
            f"(sur {len(macs)} régime(s) exécutés)."
        )
    return "\n".join(lines) + "\n"


def render_stream_report(streams: list[dict]) -> str:
    if not streams:
        return ""
    lines = [
        "## Convolution en streaming (`stream_bench`, `ConvStreaming`)",
        "",
        "| Régime | W | Noyau | cycles/ligne | µs/ligne | cycles/MAC | FPS@120p | FPS@240p | FPS@480p | FPS@720p |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in streams:
        lines.append(
            f"| {s['name']} | {s['W']} | {s['KH']}×{s['KW']} | {s['cycles_per_row']} | "
            f"{s['us_per_row']:.2f} | {s['cycles_per_mac']:.2f} | "
            f"{s['fps_120']:.0f} | {s['fps_240']:.0f} | {s['fps_480']:.0f} | {s['fps_720']:.0f} |"
        )

    lines += ["", "**Synthèse** — plafond FPS compute-bound (ignore acquisition capteur/DMA) :"]
    for height_key, label in [("fps_120", "120p"), ("fps_240", "240p"), ("fps_480", "480p"), ("fps_720", "720p")]:
        sustain_100 = [s for s in streams if s[height_key] >= 100.0]
        if sustain_100:
            widths = ", ".join(f"{s['name']} ({s[height_key]:.0f} fps)" for s in sustain_100)
            lines.append(f"- **{label}** : tient les 100Hz avec {widths}.")
        else:
            best = max(streams, key=lambda s: s[height_key])
            lines.append(
                f"- **{label}** : aucun régime mesuré ne tient 100Hz "
                f"(meilleur : {best['name']} à {best[height_key]:.0f} fps)."
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", type=Path, help="fichier(s) log RTT à analyser")
    ap.add_argument("--out", type=Path, default=None, help="écrit aussi le rapport dans ce fichier .md")
    args = ap.parse_args()

    all_regimes: list[dict] = []
    all_streams: list[dict] = []
    for log_path in args.logs:
        if not log_path.exists():
            print(f"introuvable, ignoré : {log_path}", file=sys.stderr)
            continue
        r, s = parse_log(log_path)
        all_regimes += r
        all_streams += s

    if not all_regimes and not all_streams:
        print("Aucune ligne REGIME| ou STREAM| trouvée dans les logs fournis.", file=sys.stderr)
        return 1

    parts = ["# Analyse automatique — budget conv & FPS streaming", ""]
    if all_regimes:
        parts.append(render_regime_report(all_regimes))
    if all_streams:
        parts.append(render_stream_report(all_streams))
    report = "\n".join(parts)

    print(report)
    if args.out:
        args.out.write_text(report)
        print(f"\n(rapport aussi écrit dans {args.out})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
