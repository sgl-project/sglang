#!/bin/bash

# Script to update reference videos using videos from generated_videos directory
# Both directories should exist in the same directory as this script

set -e  # Exit on any error

# Define directory paths
GENERATED_DIR="generated_videos"
REFERENCE_DIR="L40S_reference_videos"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting reference video update...${NC}"

# Check if generated_videos directory exists
if [ ! -d "$GENERATED_DIR" ]; then
    echo -e "${RED}Error: $GENERATED_DIR directory not found!${NC}"
    exit 1
fi

# Check if reference_videos directory exists
if [ ! -d "$REFERENCE_DIR" ]; then
    echo -e "${RED}Error: $REFERENCE_DIR directory not found!${NC}"
    exit 1
fi

# Function to copy videos recursively
copy_videos() {
    local src_dir="$1"
    local dst_dir="$2"

    # Find all video files in the source directory
    find "$src_dir" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" -o -name "*.webm" -o -name "*.flv" \) | while read -r video_file; do
        # Get relative path from source directory
        relative_path="${video_file#$src_dir/}"

        # Construct destination path
        dst_file="$dst_dir/$relative_path"

        # Create destination directory if it doesn't exist
        dst_file_dir=$(dirname "$dst_file")
        mkdir -p "$dst_file_dir"

        # Copy the video file
        echo -e "${GREEN}Copying: $relative_path${NC}"
        cp "$video_file" "$dst_file"
    done
}

# Perform the copy operation
echo -e "${YELLOW}Copying videos from $GENERATED_DIR to $REFERENCE_DIR...${NC}"
copy_videos "$GENERATED_DIR" "$REFERENCE_DIR"

echo -e "${GREEN}Reference videos updated successfully!${NC}"

# Show summary
video_count=$(find "$GENERATED_DIR" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" -o -name "*.webm" -o -name "*.flv" \) | wc -l)
echo -e "${YELLOW}Total videos processed: $video_count${NC}"
