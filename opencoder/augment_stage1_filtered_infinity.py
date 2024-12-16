from datasets import load_dataset
import jsonlines
from openai import OpenAI
from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor, wait
import threading
from tenacity import retry, wait_exponential, stop_after_attempt
import time
import random
from tqdm import tqdm

NUM_WORKERS = 30
SKIP=8550

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_ENDPOINT")
model = "gemini-2.0-flash-exp"

# Initialize OpenAI client
client = OpenAI(api_key=api_key, base_url=base_url)

@retry(wait=wait_exponential(multiplier=2, min=2, max=120), stop=stop_after_attempt(500))
def generate_openai_response(messages, max_tokens=8000, response_format=None):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=1.0,
            response_format=response_format
        )
        return response
    except Exception as e:
        print(f"OpenAI API error: {e}")
        raise

# Load dataset as an iterator
dataset = load_dataset("OpenCoder-LLM/opc-sft-stage1", "filtered_infinity_instruct")["train"]
total_samples = len(dataset)
dataset = iter(dataset.skip(SKIP))
output_file = "opc-sft-stage1.filtered_infinity_instruct.augmented.jsonl"
lock = threading.Lock()

from multiprocessing import Value
processed_samples = Value('i', 0)
writer = jsonlines.open(output_file, mode='a')

def worker(progress_bar):
    while True:
        try:
            sample = next(dataset)
        except StopIteration:
            return

        try:
            response = generate_openai_response([{"role": "user", "content": sample["instruction"]}])
            result = {
                "instruction": sample["instruction"],
                "output": sample["output"],  # Keep the original output
                "augmented": response.choices[0].message.content,
                "tag": "filtered-infinity-instruct-augmented"
            }
            
            with lock:  # Using the existing lock for file writing
                writer.write(result)
                
            with processed_samples.get_lock():
                processed_samples.value += 1
            progress_bar.update(1)
            
        except Exception as e:
            print(f"Error processing task {e}")
            continue

def main():
    progress_bar = tqdm(total=total_samples - SKIP, desc="Processing samples")
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(worker, progress_bar) for _ in range(NUM_WORKERS)]
        wait(futures) 
    
    progress_bar.close()
    writer.close()

if __name__ == "__main__":
    main()
