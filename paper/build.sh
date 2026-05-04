#!/usr/bin/env bash
# Build Paper 2 and produce a versioned copy alongside main.pdf.
set -euo pipefail

VERSION="v2"
OUT_NAME="Khan_2026_Volatility_Forecasting_HorseRace_${VERSION}.pdf"

cd "$(dirname "$0")"
tectonic main.tex
cp main.pdf "${OUT_NAME}"
echo "Built: main.pdf"
echo "Versioned copy: ${OUT_NAME}"
