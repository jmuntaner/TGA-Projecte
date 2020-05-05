#!/bin/bash
export PATH=/Soft/cuda/9.0.176/bin:$PATH
nvprof --unified-memory-profiling off ./cudacode.exe
