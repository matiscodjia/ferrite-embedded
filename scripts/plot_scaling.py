#!/usr/bin/env python3
"""Trace les deux axes (canaux, résolution) à partir des CSV produits par
analyze.py --csv-dir (regime.csv, scale.csv). À lancer sur la machine hôte
(pas la cible) : dépendances matplotlib + numpy.

Usage:
    python3 scripts/plot_scaling.py scripts/csv --out scripts/plots
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Palette catégorielle validée (voir dataviz skill) : bleu, orange, aqua, jaune.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_SECONDARY, MUTED, GRID, SURFACE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb",
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _num(row: dict, key: str) -> float:
    return float(row[key])


def plot_space_vs_resolution_channels(regime_rows: list[dict], out_dir: Path) -> None:
    """Axe 1 (canaux) x Axe 2 (résolution) sur l'espace (RAM) : deux séries
    (GRIS=C1, RGB=C3) issues de la grille K=1 propre (grid_c1_*/grid_c3_*),
    RAM_BYTES tracé même pour les régimes SKIP (calculé à la compilation,
    valable qu'on ait pu l'exécuter ou non)."""
    series = {1: ("GRIS (C=1)", BLUE), 3: ("RGB (C=3)", ORANGE)}
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for c, (label, color) in series.items():
        rows = [r for r in regime_rows if r["name"].startswith(f"grid_c{c}_")]
        rows.sort(key=lambda r: _num(r, "H"))
        pixels = np.array([_num(r, "H") * _num(r, "W") for r in rows])
        ram = np.array([_num(r, "ram_bytes") for r in rows])
        ok = np.array([r["status"] == "OK" for r in rows])

        ax.plot(pixels, ram, "-", color=color, linewidth=2, zorder=2)
        ax.scatter(pixels[ok], ram[ok], s=42, color=color, zorder=3, label=f"{label} — mesuré (OK)")
        ax.scatter(pixels[~ok], ram[~ok], s=42, facecolors="none", edgecolors=color,
                   linewidths=1.6, zorder=3, label=f"{label} — SKIP (calculé, RAM insuffisante)")

        slope, intercept = np.polyfit(pixels, ram, 1)
        fit_x = np.array([pixels.min(), pixels.max()])
        ax.plot(fit_x, slope * fit_x + intercept, "--", color=color, linewidth=1.2, alpha=0.6, zorder=1)
        r2 = 1 - np.sum((ram - (slope * pixels + intercept)) ** 2) / np.sum((ram - ram.mean()) ** 2)
        print(f"[espace] C={c}: RAM ~= {slope:.3f} * pixels + {intercept:.0f}  (R^2={r2:.4f})")

    ax.axhline(75_000, color=MUTED, linewidth=1, linestyle=":", zorder=1)
    ax.text(ax.get_xlim()[1] * 0.98, 75_000, " budget sécurité (75 Ko)", color=MUTED,
            fontsize=9, ha="right", va="bottom")
    ax.axhline(131_072, color=INK_SECONDARY, linewidth=1, linestyle=":", zorder=1)
    ax.text(ax.get_xlim()[1] * 0.98, 131_072, " RAM totale (128 Ko)", color=INK_SECONDARY,
            fontsize=9, ha="right", va="bottom")

    ax.set_xlabel("Résolution (pixels par image, H×W)")
    ax.set_ylabel("Espace occupé — tenseurs sur pile (octets)")
    ax.set_title("Espace vs résolution, par nombre de canaux (K=1, noyau 3×3, plein-cadre)")
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "space_vs_resolution_channels.png", dpi=160)
    plt.close(fig)


def plot_fps_native_resolution(scale_rows: list[dict], out_dir: Path) -> None:
    """Axe 1 x Axe 2 sur la vitesse : FPS mesuré à la résolution *native*
    de chaque largeur testée (w160->120p, w320->240p, w640->480p,
    w1280->720p), pour chaque nombre de canaux -- le graphe qui répond
    directement à "FPS en fonction des deux axes"."""
    anchors = [("120p\n160×120", 160, "fps_120"), ("240p\n320×240", 320, "fps_240"),
               ("480p\n640×480", 640, "fps_480"), ("720p\n1280×720", 1280, "fps_720")]
    channels = [1, 2, 3, 4]
    colors = {1: BLUE, 2: AQUA, 3: ORANGE, 4: YELLOW}
    labels = {1: "GRIS (C=1)", 2: "C=2 (contrôle)", 3: "RGB (C=3)", 4: "C=4 (contrôle)"}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    n_groups = len(anchors)
    n_bars = len(channels)
    bar_w = 0.8 / n_bars
    x = np.arange(n_groups)

    for i, c in enumerate(channels):
        vals = []
        for label, w, fps_key in anchors:
            row = next(r for r in scale_rows if r["C"] == str(c) and r["W"] == str(w))
            vals.append(_num(row, fps_key))
        offset = (i - (n_bars - 1) / 2) * bar_w
        bars = ax.bar(x + offset, vals, width=bar_w * 0.92, color=colors[c], label=labels[c])
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.0f}", ha="center",
                    va="bottom", fontsize=7.5, color=INK_SECONDARY)

    ax.axhline(100, color=MUTED, linewidth=1, linestyle=":", zorder=0)
    ax.text(n_groups - 0.5, 100, " cible 100 Hz", color=MUTED, fontsize=9, ha="right", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels([a[0] for a in anchors])
    ax.set_ylabel("FPS plafond compute-bound (ConvStreaming, noyau 3×3)")
    ax.set_title("Vitesse (FPS) vs résolution native ET nombre de canaux")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "fps_vs_resolution_channels.png", dpi=160)
    plt.close(fig)


def plot_axis_isolation(scale_rows: list[dict], out_dir: Path) -> None:
    """Deux panneaux qui isolent chaque axe : (gauche) coût/ligne vs largeur
    à C=1 fixe ; (droite) coût/ligne vs canaux à 4 largeurs fixes -- pour
    vérifier que la pente canaux ne dépend pas de la résolution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    c1_rows = sorted((r for r in scale_rows if r["C"] == "1" and r["name"].startswith("c1_w")),
                      key=lambda r: _num(r, "W"))
    w = np.array([_num(r, "W") for r in c1_rows])
    us = np.array([_num(r, "us_per_row") for r in c1_rows])
    ax1.plot(w, us, "-o", color=BLUE, markersize=5)
    slope, intercept = np.polyfit(w, us, 1)
    r2 = 1 - np.sum((us - (slope * w + intercept)) ** 2) / np.sum((us - us.mean()) ** 2)
    print(f"[vitesse] axe resolution (C=1): us/ligne ~= {slope:.4f} * W + {intercept:.2f} (R^2={r2:.4f})")
    ax1.set_xlabel("Largeur W (px)")
    ax1.set_ylabel("Temps par ligne (µs)")
    ax1.set_title("Axe résolution (C=1 fixe)")

    widths = [160, 320, 640, 1280]
    colors = {160: BLUE, 320: AQUA, 640: ORANGE, 1280: YELLOW}
    for wv in widths:
        rows = sorted((r for r in scale_rows if r["W"] == str(wv)), key=lambda r: _num(r, "C"))
        c = np.array([_num(r, "C") for r in rows])
        us_ = np.array([_num(r, "us_per_row") for r in rows])
        ax2.plot(c, us_, "-o", color=colors[wv], markersize=5, label=f"W={wv}")
        slope, intercept = np.polyfit(c, us_, 1)
        r2 = 1 - np.sum((us_ - (slope * c + intercept)) ** 2) / np.sum((us_ - us_.mean()) ** 2)
        print(f"[vitesse] axe canaux (W={wv}): us/ligne ~= {slope:.4f} * C + {intercept:.2f} (R^2={r2:.4f})")
    ax2.set_xlabel("Nombre de canaux C")
    ax2.set_xticks([1, 2, 3, 4])
    ax2.set_title("Axe canaux (W fixe, 4 résolutions)")
    ax2.legend(fontsize=8, frameon=False)

    fig.suptitle("Isolation des deux axes sur le coût par ligne (ConvStreaming)")
    fig.tight_layout()
    fig.savefig(out_dir / "speed_axis_isolation.png", dpi=160)
    plt.close(fig)


def plot_fps_direct_vs_resolution(regime_rows: list[dict], scale_rows: list[dict], out_dir: Path) -> None:
    """FPS du chemin *direct* (plein-cadre, tensordot_3/im2col, une seule
    passe par image) vs résolution carrée H=W, par canal -- avec le mur RAM
    annoté (dernier point OK / premier SKIP). Superpose le seul point où une
    comparaison à résolution égale avec le streaming est possible (C=1,
    W=80) pour visualiser concrètement "plus de FPS, mais stoppe plus tôt".
    """
    series = {1: ("GRIS (C=1)", BLUE), 3: ("RGB (C=3)", ORANGE)}
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for c, (label, color) in series.items():
        rows = sorted((r for r in regime_rows if r["name"].startswith(f"grid_c{c}_")),
                      key=lambda r: _num(r, "H"))
        ok_rows = [r for r in rows if r["status"] == "OK"]
        skip_rows = [r for r in rows if r["status"] == "SKIP"]

        h_ok = np.array([_num(r, "H") for r in ok_rows])
        fps_ok = np.array([1e6 / _num(r, "time_us") for r in ok_rows])
        ax.plot(h_ok, fps_ok, "-o", color=color, markersize=5, label=f"{label} — direct (mesuré)")

        if skip_rows:
            wall_h = _num(skip_rows[0], "H")
            ax.axvline(wall_h, color=color, linestyle=":", linewidth=1.3, alpha=0.7)
            ax.text(wall_h, fps_ok.max() * 1.02,
                    f" mur RAM {label}\n {int(wall_h)}×{int(wall_h)}", color=color,
                    fontsize=8, ha="left", va="bottom")

    # Le seul point comparable à résolution strictement égale : C=1, 80x80.
    stream_80 = next((r for r in scale_rows if r["name"] == "c1_w80"), None)
    if stream_80:
        cyc = _num(stream_80, "cycles_per_row")
        fps_stream_80 = (168_000_000.0) / (80 * cyc)
        ax.scatter([80], [fps_stream_80], marker="*", s=180, color=INK,
                   zorder=5, label="streaming, même résolution (80×80, C=1)")

    ax.set_xlabel("Côté de l'image (H=W, px)")
    ax.set_ylabel("FPS (chemin direct, plein-cadre)")
    ax.set_title("FPS du chemin direct vs résolution — plus rapide, mais mur RAM précoce")
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "fps_direct_vs_resolution_wall.png", dpi=160)
    plt.close(fig)


def plot_cycles_per_mac_direct_vs_streaming(regime_rows: list[dict], scale_rows: list[dict], out_dir: Path) -> None:
    """cycles/MAC (plus bas = plus efficace) : direct (grid 64×64, seule
    résolution mesurée pour C=1 ET C=3) vs streaming (W=160, mesuré pour
    C=1..4) -- isole l'effet "efficacité par MAC", indépendamment de la
    résolution exacte, pour expliquer pourquoi le direct fait plus de FPS."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    direct = {int(r["C"]): _num(r, "cycles_per_mac")
              for r in regime_rows if r["name"] in ("grid_c1_064", "grid_c3_064")}
    stream = {int(r["C"]): _num(r, "cycles_per_mac")
              for r in scale_rows if r["W"] == "160"}

    cs_direct = sorted(direct)
    cs_stream = sorted(stream)
    ax.plot(cs_direct, [direct[c] for c in cs_direct], "-o", color=BLUE, markersize=7,
            label="direct (plein-cadre, 64×64)")
    ax.plot(cs_stream, [stream[c] for c in cs_stream], "-o", color=ORANGE, markersize=7,
            label="streaming (ConvStreaming, W=160)")

    ax.set_xlabel("Nombre de canaux C")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylabel("cycles / MAC (plus bas = plus efficace)")
    ax.set_title("Efficacité par MAC : direct vs streaming")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "cycles_per_mac_direct_vs_streaming.png", dpi=160)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv_dir", type=Path, help="dossier contenant regime.csv / scale.csv")
    ap.add_argument("--out", type=Path, default=Path("scripts/plots"), help="dossier de sortie des PNG")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    regime_path = args.csv_dir / "regime.csv"
    scale_path = args.csv_dir / "scale.csv"

    regime_rows = read_csv(regime_path) if regime_path.exists() else None
    scale_rows = read_csv(scale_path) if scale_path.exists() else None
    if regime_rows is None:
        print(f"absent, ignoré : {regime_path}")
    if scale_rows is None:
        print(f"absent, ignoré : {scale_path}")

    if regime_rows:
        plot_space_vs_resolution_channels(regime_rows, args.out)

    if scale_rows:
        plot_fps_native_resolution(scale_rows, args.out)
        plot_axis_isolation(scale_rows, args.out)

    if regime_rows and scale_rows:
        plot_fps_direct_vs_resolution(regime_rows, scale_rows, args.out)
        plot_cycles_per_mac_direct_vs_streaming(regime_rows, scale_rows, args.out)

    print(f"\nGraphiques écrits dans {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
