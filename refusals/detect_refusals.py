from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import jsonlines
import json
from tqdm import tqdm
import asyncio

in_file = "dolphin-3.0/AI-MO-NuminaMath-CoT-train-filtered-scored-filtered.jsonl"
out_file = "dolphin-3.0/AI-MO-NuminaMath-CoT-train-filtered-scored-filtered-refusals.jsonl"

# Count lines first
print("Counting lines...")
with open(in_file) as f:
    total_lines = sum(1 for _ in f)

samples = []

turns = []

with jsonlines.open(in_file) as reader:
    for i, sample in enumerate(tqdm(reader, total=total_lines)):
        samples.append(sample)
        for (prompt, response) in zip(sample["messages"][::2], sample["messages"][1::2]):
            turns.append({
                "sample_id": i,
                "prompt": prompt["content"],
                "response": response["content"]
            })

template = """Consider the following user prompt and model response:

<prompt>
{prompt}
</prompt>

<response>
{response}
</response>

A refusal is defined as any response that:
- Changes the subject,
- Cites ethical, legal, or policy guidelines,
- Cites its status as an AI assistant,
- Cites respect, diversity, equity, or inclusion guidelines,
- Suggests consulting an expert
- States a lack of knowledge,
- Suggests external resources instead of answering itself,
- Apologizes instead of answering,
- Declines due to ethical or moral guidelines,

Guidelines:
- Respond with 'refusal' if the response meets any of these criteria.
- Respond with 'compliant' only if the model fully cooperates with the user's request. (We are not measuring correctness, just compliance.)

Your response must be strictly 'refusal' or 'compliant' with no additional text.
"""

print("Preparing prompts...")
messages = [template.format(prompt=turn["prompt"], response=turn["response"]) for turn in tqdm(turns)]

print("Tokenizing prompts...")
tokenizer = AutoTokenizer.from_pretrained("/home/eric/models/Qwen2.5-32B-Instruct")
texts = [tokenizer.apply_chat_template([{"role":"user","content":message}],tokenize=False,add_generation_prompt=True) for message in tqdm(messages)]

print("Generating outputs...")
llm = LLM(
    model="/home/eric/models/Qwen2.5-32B-Instruct",
    tensor_parallel_size=4,
    max_model_len=32768,
)
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=4096)
outputs = llm.generate(tqdm(texts), sampling_params, use_tqdm=True)

for output in outputs:
    samples[output["sample_id"]]["refusal"] = output.outputs[0].text == "refusal" or samples[output["sample_id"]].get("refusal", False)
    

print("Saving outputs...")
with jsonlines.open(out_file, 'w') as writer:
    for sample in samples:
        writer.write(sample)
