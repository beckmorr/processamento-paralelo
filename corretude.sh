#!/bin/bash
set -euo pipefail
export LC_NUMERIC="C"

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "=== RUNNING CHECKS: seq, mpi, omp, cuda ==="

# ------------------------------ Sequencial ------------------------------
echo "\n--- SEQ ---"
if [ -f ./gauss_blur_seq ]; then
  echo "Using existing ./gauss_blur_seq"
else
  echo "Compiling gauss_blur_seq.c..."
  gcc -O2 gauss_blur_seq.c -lm -o gauss_blur_seq
fi
./gauss_blur_seq --check

# ------------------------------ OpenMP ------------------------------
echo "\n--- OMP (2 threads) ---"
if [ -f ./gauss_blur_omp ]; then
  echo "Using existing ./gauss_blur_omp"
else
  echo "Compiling gauss_blur_omp.c with OpenMP support..."
  gcc -O2 -fopenmp gauss_blur_omp.c -lm -o gauss_blur_omp
fi
OMP_NUM_THREADS=2 ./gauss_blur_omp --check

# ------------------------------ MPI ------------------------------
echo "\n--- MPI (2 processes) ---"
if command -v mpicc >/dev/null 2>&1; then
  if [ -f ./gauss_blur_mpi ]; then
    echo "Using existing ./gauss_blur_mpi"
  else
    echo "Compiling gauss_blur_mpi.c with mpicc..."
    mpicc -O2 gauss_blur_mpi.c -lm -o gauss_blur_mpi
  fi
  if command -v mpirun >/dev/null 2>&1; then
    mpirun -np 2 ./gauss_blur_mpi --check
  else
    echo "mpirun not found; skipping MPI run"
  fi
else
  echo "mpicc not found; skipping MPI compile/run"
fi

# ------------------------------ CUDA ------------------------------
echo "\n--- CUDA ---"
if command -v nvcc >/dev/null 2>&1; then
  if [ -f ./gauss_blur_cuda ]; then
    echo "Using existing ./gauss_blur_cuda"
  else
    echo "Compiling gauss_blur_cuda.cu with nvcc..."
    nvcc -O2 gauss_blur_cuda.cu -lm -o gauss_blur_cuda
  fi
  ./gauss_blur_cuda --check
else
  echo "nvcc not found; skipping CUDA compile/run"
fi

echo "\n=== ALL CHECKS FINISHED ==="
