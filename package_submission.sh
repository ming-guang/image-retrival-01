#!/usr/bin/env bash
# Assembles the submission folder <MSSV>/{Source,Release,Docs} from the working
# tree and zips it. Paths resolve relative to this script, so it runs from
# anywhere. Options:
#   --no-build   skip recompiling the engine/GUI (use whatever is in releases/)
set -euo pipefail

MSSV=20127298
NAME="${MSSV}_DOAN"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUB="$ROOT/submission/$NAME"
ZIP="$ROOT/$NAME.zip"
BUILD=1
[ "${1:-}" = "--no-build" ] && BUILD=0

if [ "$BUILD" = 1 ]; then
  echo "==> Building engine + GUI (make -C sources)"
  make -C "$ROOT/sources" >/dev/null
fi

echo "==> Assembling $SUB"
rm -rf "$ROOT/submission"   # drop any older layout (e.g. plain MSSV folder)
mkdir -p "$SUB/Source/sources/src" "$SUB/Source/sources/gui" \
         "$SUB/Source/sources/benchmark" "$SUB/Release" "$SUB/Docs"

# Source: compile inputs only (no *.o, no binaries, no hidden dirs).
cp "$ROOT"/sources/src/*.cpp "$ROOT"/sources/src/*.h "$SUB/Source/sources/src/"
cp "$ROOT"/sources/Makefile "$SUB/Source/sources/"
cp "$ROOT"/sources/build_bundle.py "$SUB/Source/sources/"
cp "$ROOT"/sources/gui/image_retrieval_gui.py "$SUB/Source/sources/gui/"
cp "$ROOT"/sources/benchmark/benchmark.py "$SUB/Source/sources/benchmark/"
cp "$ROOT"/flake.nix "$SUB/Source/"

# Release: compiled program (OpenCV >3.0 -> no dll needed).
cp "$ROOT"/releases/image-retrieval "$ROOT"/releases/image-retrieval-gui \
   "$SUB/Release/"

# Docs: the report plus the evaluation results, if present.
cp "$ROOT"/documents/report.pdf "$SUB/Docs/"
if [ -d "$ROOT/results" ]; then
  mkdir -p "$SUB/Docs/results"
  cp "$ROOT"/results/* "$SUB/Docs/results/" 2>/dev/null || true
fi

echo "==> Zipping -> $ZIP"
rm -f "$ZIP" "$ROOT/$MSSV.zip"   # remove old-named zip too
( cd "$ROOT/submission" && zip -rq "$ZIP" "$NAME" )

echo "Done. Archive contents:"
unzip -l "$ZIP"
