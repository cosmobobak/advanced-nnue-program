#!/usr/bin/env python3
"""
Convert correlations.txt from Python literal format [[1, 2], [3, 4]] to numpy .npy binary format.
This provides much faster loading performance for large matrices.
"""

import ast
import numpy as np

def convert_correlations_txt_to_npy(input_file='correlations.txt', output_file='correlations.npy'):
    """Convert correlations.txt to correlations.npy for faster loading."""
    
    # Load the Python literal format
    with open(input_file, 'r') as f:
        matrix_list = ast.literal_eval(f.read())
    
    # Convert to numpy array
    matrix = np.array(matrix_list, dtype=int)
    
    # Save as binary format
    np.save(output_file, matrix)
    
    print(f'Converted matrix with shape {matrix.shape} from {input_file} to {output_file}')

if __name__ == '__main__':
    convert_correlations_txt_to_npy()