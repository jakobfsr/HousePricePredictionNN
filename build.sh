#!/usr/bin/env bash
set -e

# Prüfen, ob mindestens drei Argumente übergeben wurden
if [ $# -lt 3 ]; then
    echo "Usage: $0 <cross-val|train> <batch_size> <num_threads>"
    exit 1
fi

# Build-Verzeichnis vorbereiten
if [ -d "build" ]; then
    rm -rf build
fi

mkdir build
cd build

# CMake und Make ausführen
cmake ..
make -j4

# Programm mit den Argumenten starten
./nn_example "$1" "$2" "$3"
