#!/usr/bin/env python3
"""
Convert .npy float32 vectors to DiskANN .fbin format.
Usage: python3 npy_to_fbin.py input.npy output.fbin
"""
import sys
import struct
import numpy as np

def npy_to_fbin(input_path, output_path):
    data = np.load(input_path)
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    
    npts, dim = data.shape
    print(f"Converting {input_path}: {npts} points x {dim} dims -> {output_path}")
    
    with open(output_path, 'wb') as f:
        f.write(struct.pack('I', npts))  # uint32 npts
        f.write(struct.pack('I', dim))   # uint32 dim
        f.write(data.tobytes())
    
    print(f"Written {output_path} ({npts * dim * 4 + 8} bytes)")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} input.npy output.fbin")
        sys.exit(1)
    npy_to_fbin(sys.argv[1], sys.argv[2])
