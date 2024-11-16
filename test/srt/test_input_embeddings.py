import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import tempfile
import os
import subprocess
# Load the model and tokenizer
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
# List of texts
texts = [
    "The capital of France is",
    "What is the best time of year to visit Japan for cherry blossoms?",
    "How do I navigate the public transportation system in Tokyo?",
    "What are some must-try traditional Japanese foods and where can I find them?",
    "Is it necessary to learn Japanese to travel around Japan, or can I get by with English?",
    "What are the most popular cultural sites and historical landmarks to visit in Kyoto?",
    "Are there any specific customs or etiquette I should be aware of when visiting temples and shrines?",
    "What types of accommodations are available in Japan, and how do they differ from Western hotels?",
    "How can I stay connected while traveling in Japan? Should I rent a pocket Wi-Fi or buy a SIM card?",
    "What are some unique experiences or activities I shouldn’t miss while in Japan?",
    "How do I handle cash and credit cards in Japan, and are there places where cards are not accepted?",
    "When is the best time to see cherry blossoms in Japan?"
]
# Create a list to store the results
results = []
# Iterate over each text, tokenize and get embeddings
for text in texts:
    # Tokenize the input text
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    # Get embeddings from the model's input embeddings
    embeddings = model.get_input_embeddings()(input_ids)
    # Convert embeddings to a list of lists for saving to JSON
    embeddings_list = embeddings.squeeze().tolist()  # squeeze to remove batch dimension
    # Append the result as a dictionary with 'text' and 'embeddings'
    results.append({
        "text": text,
        "embeddings": embeddings_list
    })
# Write results to a JSON file
output_file = "./input_embeddings.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)
print(f"Embeddings successfully saved to {output_file}")



# Test written with Qwen2.5 in mind, 
# if wanting to test with other model 
# please generate new embeddings by 
# altering embedding.py
# Load the embeddings JSON file
with open('./input_embeddings.json', 'r') as f:
    embeddings_data = json.load(f)
# Open the file to save the comparison results
comparison_file = "comparison_responses.txt"
with open(comparison_file, 'w') as f_output:
    f_output.write("Text vs Embedding Response Comparison\n")
    f_output.write("=" * 40 + "\n")
# Loop through each entry in the JSON file
for entry in embeddings_data:
    text = entry["text"]
    input_embeds = entry["embeddings"]
    # Prepare the payload for sending text
    text_input_data = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "text": text,  # Using text for the first request
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 50
        }
    }
    # Create a temporary file to store the text-based JSON data
    with tempfile.NamedTemporaryFile(delete=True, mode='w', suffix='.json') as temp_file:
        json.dump(text_input_data, temp_file)
        temp_file.flush()  # Ensure the data is written before using the file
        # Curl command for text
        text_curl_command = [
            "curl",
            "-X", "POST",
            "http://127.0.0.1:30000/generate",  # Your API endpoint
            "-H", "Content-Type: application/json",
            "--data-binary", f"@{temp_file.name}"  # Use the temporary file for input
        ]
        # Execute the curl command for text input
        try:
            text_result = subprocess.run(text_curl_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if text_result.returncode == 0:
                text_response = text_result.stdout.strip()
            else:
                text_response = f"Error: {text_result.stderr.strip()}"
        except Exception as e:
            text_response = f"Failed to send the request for text: {e}"
    # Prepare the payload for sending input embeddings
    embed_input_data = {
        "model": "Qwen/Qwen2.5-0.5B",
        "input_embeds": input_embeds,  # Using input embeddings for the second request
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 50
        }
    }
    # Create a temporary file to store the embedding-based JSON data
    with tempfile.NamedTemporaryFile(delete=True, mode='w', suffix='.json') as temp_file:
        json.dump(embed_input_data, temp_file)
        temp_file.flush()  # Ensure the data is written before using the file
        # Curl command for embeddings
        embed_curl_command = [
            "curl",
            "-X", "POST",
            "http://127.0.0.1:30000/generate",  # Your API endpoint
            "-H", "Content-Type: application/json",
            "--data-binary", f"@{temp_file.name}"  # Use the temporary file for input
        ]
        # Execute the curl command for embedding input
        try:
            embed_result = subprocess.run(embed_curl_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if embed_result.returncode == 0:
                embed_response = embed_result.stdout.strip()
            else:
                embed_response = f"Error: {embed_result.stderr.strip()}"
        except Exception as e:
            embed_response = f"Failed to send the request for embeddings: {e}"
    # Compare the two responses
    comparison = (
        f"Input Text: {text}\n\n"
        f"Text-based response:\n{text_response}\n\n"
        f"Embedding-based response:\n{embed_response}\n"
        f"{'-' * 80}\n"
    )
    # Append the comparison result to the text file
    with open(comparison_file, 'a') as f_output:
        f_output.write(comparison)
    # Print to console for reference
    print(comparison)
print(f"All comparisons have been saved to {comparison_file}.")