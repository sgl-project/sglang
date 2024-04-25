# from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
from datetime import datetime
import hashlib
from PIL import Image
import io, os

import sglang as sgl
import time

import cv2

from sglang import function, system, user, assistant, gen, set_default_backend, RuntimeEndpoint


@sgl.function
def video_qa(s, num_frames, video_path, question):
    s += sgl.user(sgl.video(video_path,num_frames) + question)
    s += sgl.assistant(sgl.gen("answer"))

def single_video(path, prompt, num_frames=16):

    def post_process(answer):
        return state["answer"].replace("image","video")

    state = video_qa.run(
        num_frames=num_frames,
        video_path=path,
        question=prompt,
        temperature=0.0,
        max_new_tokens=1024,
    )
    
    return post_process(state["answer"])

def single_image(path, prompt, num_frames=16):
    state = video_qa.run(
        num_frames=num_frames,
        video_path=path,
        question=prompt,
        temperature=0.0,
        max_new_tokens=1024,
    )
    return state["answer"]



# app = Flask(__name__)

# Ensure model is in evaluation mode
prompt_txt_path = "./user_logs/prompts.txt"
multimodal_folder_path = "./user_logs"

if not os.path.exists(multimodal_folder_path):
    os.makedirs(multimodal_folder_path)

runtime = RuntimeEndpoint("http://localhost:30000")

sgl.set_default_backend(runtime)
# print(f"chat template: {runtime.endpoint.chat_template.name}")

def save_image_unique(pil_image, directory=multimodal_folder_path):
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Convert the PIL Image into a bytes object
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()

    # Compute the hash of the image data
    hasher = hashlib.sha256()
    hasher.update(img_byte_arr)
    hash_hex = hasher.hexdigest()

    # Create a file name with the hash value
    file_name = f"{hash_hex}.png"
    file_path = os.path.join(directory, file_name)

    # Check if a file with the same name exists
    if os.path.isfile(file_path):
        print(f"Image already exists with the name: {file_name}")
    else:
        # If the file does not exist, save the image
        with open(file_path, "wb") as new_file:
            new_file.write(img_byte_arr)
        print(f"Image saved with the name: {file_name}")

    return file_path


def save_video_unique(video_capture, directory=multimodal_folder_path):
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Read video frames and append them into a byte array
    video_byte_arr = io.BytesIO()
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        # Convert frame to bytes and append to video_byte_arr
        is_success, buffer = cv2.imencode(".jpg", frame)
        video_byte_arr.write(buffer.tobytes())

    # Compute the hash of the video data
    video_bytes = video_byte_arr.getvalue()
    hasher = hashlib.sha256()
    hasher.update(video_bytes)
    hash_hex = hasher.hexdigest()

    # Create a file name with the hash value
    file_name = f"{hash_hex}.mp4"
    file_path = os.path.join(directory, file_name)

    # Check if a file with the same name exists
    if os.path.isfile(file_path):
        print(f"Video already exists with the name: {file_name}")
    else:
        # If the file does not exist, write the video data to a new file
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec and create VideoWriter object
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        frame_size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(file_path, fourcc, fps, frame_size)

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            out.write(frame)
        out.release()  # Release the video writer object
        print(f"Video saved with the name: {file_name}")

    return file_path


# Define endpoint
# @app.route("/app/otter", methods=["POST"])
def process_image_and_prompt():
    start_time = datetime.now()
    # # Parse request data
    # data = request.get_json()
    # query_content = data["content"][0]
    # if "image" not in query_content:
    #     return jsonify({"error": "Missing Image"}), 400
    # elif "prompt" not in query_content:
    #     return jsonify({"error": "Missing Prompt"}), 400

    # # Decode the image
    # image_data = query_content["image"]
    # image = Image.open(BytesIO(base64.b64decode(image_data)))
    formated_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
    # path = save_image_unique(image)

    # # Decode the video
    # video_data = query_content["video"]
    # video = cv2.VideoCapture(BytesIO(base64.b64decode(video_data)))
    video = cv2.VideoCapture("./Q98Z4OTh8RwmDonc.mp4")
    path = save_video_unique(video)
    video.release()

    # prompt = query_content["prompt"]
    prompt = "please describe the video content"


    print(path)

    # Initialize response
    response = None
    matched = False
    
    num_frames = 16

    print("Calling Llava-Next")
    # Preprocess the image and prompt, and run the model
    response = single_video(path, prompt, num_frames)

    with open(prompt_txt_path, "a") as f:
        f.write(f"*************************{formated_time}**************************" + "\n")
        f.write(f"Image/video saved to {path}" + "\n")
        f.write(f"Prompt: {prompt}" + "\n")
        f.write(f"Response: {response}" + "\n\n")

    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    # Return the response
    return jsonify({"result": response})

# Utility functions (as per the Gradio script, you can adapt the same or similar ones)
# ... (e.g., resize_to_max, pad_to_size, etc.)

if __name__ == "__main__":
    process_image_and_prompt()
    # app.run(host="0.0.0.0", port=8890)
