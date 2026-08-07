#!/usr/bin/env bash
# Flashe + capture un des bins de benchmark (regime_bench, stream_bench, conv,
# compute...) via le runner probe-rs déjà configuré dans .cargo/config.toml,
# et sauvegarde la sortie RTT brute dans scripts/logs/ pour analyze.py.
#
# Usage : scripts/run_bench.sh <bin> [args cargo supplementaires]
#   scripts/run_bench.sh regime_bench
#   scripts/run_bench.sh stream_bench
#
# probe-rs garde la session RTT ouverte tant que la carte tourne : faire
# Ctrl-C une fois la ligne "fin <bin>" affichée.
set -euo pipefail

BIN="${1:?usage: $0 <bin-name> (e.g. regime_bench, stream_bench)}"
shift || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$LOG_DIR/${BIN}-${STAMP}.log"

echo "Flash + capture de '${BIN}' -> ${LOG}"
echo "(Ctrl-C une fois que la carte a fini d'imprimer.)"
echo

cd "$REPO_ROOT"
cargo run --release --bin "$BIN" "$@" 2>&1 | tee "$LOG"

echo
echo "Log sauvegardé : ${LOG}"
echo "Analyse : python3 scripts/analyze.py ${LOG}"
