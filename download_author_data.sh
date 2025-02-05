#!/bin/bash

# arg 1: output directory to download model-specific data
# If run via container, it needs to be downloaded under the csa_data folder

OUTPUT_DIR=$1
default_data_url=https://ftp.mcs.anl.gov/pub/candle/public/improve/reproducability/DeepTTC/

# Determine the number of directories to cut from the URL
url_length=$(($(echo "$default_data_url" | awk -F'/' '{print NF}') - 4))

# Check if the data is already downloaded
if [ -f "$OUTPUT_DIR/.downloaded" ]; then 
    echo "Data present, skipping download"
elif [ ! -f "$OUTPUT_DIR/.downloading_author_data" ]; then
    touch "$OUTPUT_DIR/.downloading_author_data"
    
    # Run wget command
    wget --recursive --no-clobber -nH --cut-dirs="$url_length" --no-parent --directory-prefix="$OUTPUT_DIR" "$default_data_url"

    touch "$OUTPUT_DIR/.downloaded"
    rm "$OUTPUT_DIR/.downloading_author_data"

    # Remove index files if they exist
    find "$OUTPUT_DIR" -type f -name "*index*" -delete
else
   # Wait for other download to finish
   iteration=0
   echo "Waiting for external download"
   while [ -f "$OUTPUT_DIR/.downloading_author_data" ]; do
     iteration=$((iteration + 1))
     if [ "$iteration" -gt 10 ]; then
       # Download takes too long, exit and warn user
       echo "Check output directory, download still in progress after $iteration minutes."
       exit 1
     fi
     sleep 60
   done
fi
