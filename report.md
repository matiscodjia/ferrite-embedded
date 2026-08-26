# Analyse automatique — budget conv & FPS streaming

## Régimes de convolution (`regime_bench`)

| Régime | H×W | C | K | Noyau | MACs | RAM | cycles/MAC | temps/iter | % tick 10ms | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| rgb32_k4 | 32×32 | 3 | 4 | 3×3 | 97,200 | 27,120 o (21%) | 6.76 | 3909 µs | 39.1% | **OK** |
| gray64_k1 | 64×64 | 1 | 1 | 3×3 | 34,596 | 31,796 o (24%) | 8.69 | 1789 µs | 17.9% | **OK** |
| rgb48_k4 | 48×48 | 3 | 4 | 3×3 | 228,528 | 61,936 o (47%) | — | — | — | **SKIP (RAM)** |
| gray96_k1 | 96×96 | 1 | 1 | 3×3 | 79,524 | 72,244 o (55%) | — | — | — | **SKIP (RAM)** |
| gray64_k4 | 64×64 | 1 | 4 | 3×3 | 138,384 | 78,032 o (60%) | — | — | — | **SKIP (RAM)** |
| gray128_k1 | 128×128 | 1 | 1 | 3×3 | 142,884 | 129,076 o (98%) | — | — | — | **SKIP (RAM)** |

**Synthèse** :
- 2 régime(s) confortables (≤50% du tick), 0 serré(s) (50-100%), 0 dépassent le tick à eux seuls, 4 écartés d'office pour RAM insuffisante.
- Régime le plus exigeant qui reste confortable : **rgb32_k4** (32×32, C=3, K=4) — 39.1% du tick, 21% de la RAM.
- cycles/MAC mesuré : min 6.76, max 8.69 (sur 2 régime(s) exécutés).

## Convolution en streaming (`stream_bench`, `ConvStreaming`)

| Régime | W | Noyau | cycles/ligne | µs/ligne | cycles/MAC | FPS@120p | FPS@240p | FPS@480p | FPS@720p |
|---|---|---|---|---|---|---|---|---|---|
| w160_k3 | 160 | 3×3 | 11858 | 70.58 | 8.34 | 118 | 59 | 30 | 20 |
| w320_k3 | 320 | 3×3 | 24801 | 147.62 | 8.67 | 56 | 28 | 14 | 9 |
| w640_k3 | 640 | 3×3 | 51337 | 305.58 | 8.94 | 27 | 14 | 7 | 5 |
| w1280_k3 | 1280 | 3×3 | 90106 | 536.35 | 7.83 | 16 | 8 | 4 | 3 |
| w320_k5 | 320 | 5×5 | 67199 | 399.99 | 8.51 | 21 | 10 | 5 | 3 |

**Synthèse** — plafond FPS compute-bound (ignore acquisition capteur/DMA) :
- **120p** : tient les 100Hz avec w160_k3 (118 fps).
- **240p** : aucun régime mesuré ne tient 100Hz (meilleur : w160_k3 à 59 fps).
- **480p** : aucun régime mesuré ne tient 100Hz (meilleur : w160_k3 à 30 fps).
- **720p** : aucun régime mesuré ne tient 100Hz (meilleur : w160_k3 à 20 fps).
