import time
import requests
import json
import uuid

base_url = "http://127.0.0.1:8000/v1"

# Generate a unique reproducible prompt with ~4600 tokens
test_id = "test_turn_suite"
chunk = "In the heart of the ancient forest, towering redwoods whispered secrets to the wind as crystal streams carved paths through granite boulders. "
reps = 160
prompt_turn1 = (chunk * reps) + "\nSummarize the essence of this forest in one short sentence:"

print(f"=== GDN PREFIX CACHING + STICKY SESSIONS VERIFICATION ===")
print(f"Base URL: {base_url}")
print(f"Prompt Turn 1 Length: {len(prompt_turn1)} characters")

# Turn 1
print("\n--- TURN 1 (Initial prompt, 4k-8k tokens) ---")
t0 = time.perf_counter()
res1 = requests.post(
    f"{base_url}/completions",
    json={
        "model": "qwen3.8-27b",
        "prompt": prompt_turn1,
        "max_tokens": 16,
        "temperature": 0.0,
    }
)
t1 = time.perf_counter()
turn1_latency = t1 - t0
d1 = res1.json()
u1 = d1.get("usage", {})
t1_text = d1.get("choices", [{}])[0].get("text", "").strip()

print(f"Status: {res1.status_code}")
print(f"Latency: {turn1_latency:.3f}s")
print(f"Usage: {u1}")
print(f"Response: {t1_text}")

time.sleep(2)

# Turn 2: append follow-up to Turn 1 prompt + completion
prompt_turn2 = prompt_turn1 + "\n" + t1_text + "\nNow provide three keywords that best describe it:"

print("\n--- TURN 2 (Appended suffix, prefix cache hit expected) ---")
t2 = time.perf_counter()
res2 = requests.post(
    f"{base_url}/completions",
    json={
        "model": "qwen3.8-27b",
        "prompt": prompt_turn2,
        "max_tokens": 16,
        "temperature": 0.0,
    }
)
t3 = time.perf_counter()
turn2_latency = t3 - t2
d2 = res2.json()
u2 = d2.get("usage", {})
t2_text = d2.get("choices", [{}])[0].get("text", "").strip()

print(f"Status: {res2.status_code}")
print(f"Latency: {turn2_latency:.3f}s")
print(f"Usage: {u2}")
print(f"Response: {t2_text}")

print("\n=== SUMMARY METRICS ===")
print(f"Turn 1 Total Request Latency: {turn1_latency:.3f}s (Prompt: {u1.get('prompt_tokens')} tokens)")
print(f"Turn 2 Total Request Latency: {turn2_latency:.3f}s (Prompt: {u2.get('prompt_tokens')} tokens)")
print(f"Latency Delta: {turn1_latency - turn2_latency:.3f}s ({(1.0 - turn2_latency / turn1_latency) * 100:.1f}% reduction)")
