#!/usr/bin/env bash
set -e

# Dieses Skript erstellt ein Build-Verzeichnis,
# führt CMake aus, kompiliert das Projekt und führt es anschließend aus.

# Falls ein "build"-Verzeichnis existiert, dieses zunächst entfernen:
if [ -d "build" ]; then
    rm -rf build
fi

mkdir build
cd build
cmake ..
make -j4

# Nach erfolgreichem Build ausführen
./nn_example
