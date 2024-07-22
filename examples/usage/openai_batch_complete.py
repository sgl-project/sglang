import json
import os
import time

import openai
from openai import OpenAI


class OpenAIBatchProcessor:
    def __init__(self, api_key):
        # client = OpenAI(api_key=api_key)
        client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

        self.client = client

    def process_batch(self, input_file_path, endpoint, completion_window):

        # Upload the input file
        with open(input_file_path, "rb") as file:
            uploaded_file = self.client.files.create(file=file, purpose="batch")

        # Create the batch job
        batch_job = self.client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint=endpoint,
            completion_window=completion_window,
        )

        # Monitor the batch job status
        while batch_job.status not in ["completed", "failed", "cancelled"]:
            time.sleep(3)  # Wait for 3 seconds before checking the status again
            print(
                f"Batch job status: {batch_job.status}...trying again in 3 seconds..."
            )
            batch_job = self.client.batches.retrieve(batch_job.id)

        # Check the batch job status and errors
        if batch_job.status == "failed":
            print(f"Batch job failed with status: {batch_job.status}")
            print(f"Batch job errors: {batch_job.errors}")
            return None

        # If the batch job is completed, process the results
        if batch_job.status == "completed":

            # print result of batch job
            print("batch", batch_job.request_counts)

            result_file_id = batch_job.output_file_id
            # Retrieve the file content from the server
            file_response = self.client.files.content(result_file_id)
            result_content = file_response.read()  # Read the content of the file

            # Save the content to a local file
            result_file_name = "batch_job_complete_results.jsonl"
            with open(result_file_name, "wb") as file:
                file.write(result_content)  # Write the binary content to the file
            # Load data from the saved JSONL file
            results = []
            with open(result_file_name, "r", encoding="utf-8") as file:
                for line in file:
                    json_object = json.loads(
                        line.strip()
                    )  # Parse each line as a JSON object
                    results.append(json_object)

            return results
        else:
            print(f"Batch job failed with status: {batch_job.status}")
            return None


# Initialize the OpenAIBatchProcessor
api_key = os.environ.get("OPENAI_API_KEY")
processor = OpenAIBatchProcessor(api_key)

# Process the batch job
input_file_path = "input_complete.jsonl"
endpoint = "/v1/completions"
completion_window = "24h"

# Process the batch job
results = processor.process_batch(input_file_path, endpoint, completion_window)

# Print the results
print(results)
