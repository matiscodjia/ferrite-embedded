#!/usr/bin/env bash
# Variante automatique de run_bench.sh : flashe + capture un bin, mais coupe
# la session RTT elle-même dès que la ligne "fin <bin>" apparaît, au lieu
# d'attendre un Ctrl-C manuel -- pour enchaîner plusieurs bins sans
# supervision humaine. Timeout de secours si la carte ne finit jamais
# d'imprimer (plantage, régression qui boucle...).
#
# Usage : scripts/run_bench_auto.sh <bin> [timeout_secs=180]
set -uo pipefail

BIN="${1:?usage: $0 <bin-name> [timeout_secs]}"
TIMEOUT="${2:-180}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$LOG_DIR/${BIN}-${STAMP}.log"

echo "Flash + capture auto de '${BIN}' -> ${LOG} (timeout ${TIMEOUT}s)" >&2

cd "$REPO_ROOT"
# setsid : nouveau groupe de processus, pour pouvoir tuer proprement cargo
# ET le probe-rs qu'il lance en enfant en une seule fois.
setsid cargo run --release --bin "$BIN" >"$LOG" 2>&1 &
PGID=$!

ELAPSED=0
while (( ELAPSED < TIMEOUT * 2 )); do
    if grep -q "fin ${BIN}" "$LOG" 2>/dev/null; then
        sleep 1 # laisse la dernière ligne finir de s'écrire côté RTT
        break
    fi
    sleep 0.5
    ELAPSED=$((ELAPSED + 1))
done

kill -TERM -- -"$PGID" 2>/dev/null
sleep 0.5
kill -KILL -- -"$PGID" 2>/dev/null
wait "$PGID" 2>/dev/null

if grep -q "fin ${BIN}" "$LOG"; then
    echo "OK: '${BIN}' termine proprement, log: ${LOG}" >&2
else
    echo "ATTENTION: '${BIN}' n'a pas affiche 'fin ${BIN}' avant timeout (${TIMEOUT}s) - log partiel: ${LOG}" >&2
fi
echo "$LOG"
