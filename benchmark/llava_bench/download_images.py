import os

# Create the 'images' directory if it doesn't exist
if not os.path.exists("images"):
    os.makedirs("images")

# Base URL
base_url = "https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild/resolve/main/images/"

# Loop through image numbers
for i in range(1, 25):
    # Format the image number with leading zeros
    image_number = str(i).zfill(3)
    image_url = base_url + image_number + ".jpg"
    image_path = "images/" + image_number + ".jpg"

    # Download the image using wget
    os.system(f"wget -O {image_path} {image_url}")

print("Download complete.")
