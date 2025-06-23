#!/bin/bash

# Check if the required number of arguments is provided
if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <input_directory> <output_file> <viridithas_path> <marlinflow_utils_path> [max_jobs]"
    exit 1
fi

input_dir="$1"
output_file="$2"
viridithas_path="$3"
marlinflow_utils_path="$4"
max_jobs="${5:-$(nproc)}"
temp_dir=$(mktemp -d)

# Function to clean up temporary files
cleanup() {
    rm -rf "$temp_dir"
}

# Set up trap to clean up on exit
trap cleanup EXIT

# Function to process a single file
process_file() {
    local binpack="$1"
    local filename=$(basename "$binpack")
    local file_temp_dir="$temp_dir/$filename"
    mkdir -p "$file_temp_dir"

    local decompressed="$file_temp_dir/${filename%.bin}_decompressed.bin"
    local shuffled="$file_temp_dir/${filename%.bin}_shuffled.bin"

    "$viridithas_path" splat "$binpack" "$decompressed"
    "$marlinflow_utils_path" shuffle "$decompressed" --output "$shuffled"
    rm "$decompressed"

    # Move the shuffled file to the main temp directory
    mv "$shuffled" "$temp_dir/"
}

export -f process_file
export temp_dir
export viridithas_path
export marlinflow_utils_path

# Process all binpacks in parallel
find "$input_dir" -name "*.bin" | parallel -j "$max_jobs" process_file

# Interleave all shuffled files
shuffled_files=("$temp_dir"/*_shuffled.bin)
"$marlinflow_utils_path" interleave "${shuffled_files[@]}" --output "$output_file"

echo "Processing complete. Output file: $output_file"
