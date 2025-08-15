import spacy
import time
import numpy as np

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Define the input file path
input_file = "test_files/test_cases.txt"

print(f"--- Testing Sentence Segmentation on {input_file} ---")

# Read the file and process each line
try:
    with open(input_file, 'r') as f:
        # Read lines and strip leading/trailing whitespace
        lines = [line.strip() for line in f.readlines() if line.strip()]

    latencies = []
    total_processing_time = 0
    total_words = 0
    num_lines = len(lines)

    # Process each line from the file
    for line in lines:
        total_words += len(line.split())
        start_time = time.time()
        # Process the text with spaCy
        doc = nlp(line)
        # Force processing of the doc by iterating over sentences
        _ = [sent.text for sent in doc.sents]
        end_time = time.time()
        
        latency = (end_time - start_time) * 1000  # in milliseconds
        latencies.append(latency)
        total_processing_time += (end_time - start_time)

    # --- Performance Metrics ---
    
    # Throughput
    throughput_lps = num_lines / total_processing_time if total_processing_time > 0 else 0
    throughput_wps = total_words / total_processing_time if total_processing_time > 0 else 0
    throughput_wpms = total_words / (total_processing_time * 1000) if total_processing_time > 0 else 0
    
    # Latency Percentiles
    p50_latency = np.percentile(latencies, 50)
    p90_latency = np.percentile(latencies, 90)
    p99_latency = np.percentile(latencies, 99)
    
    # --- Display Results ---
    
    print("\n--- Performance Metrics ---")
    print(f"Total Lines Processed: {num_lines}")
    print(f"Total Words Processed: {total_words}")
    print(f"Total Processing Time: {total_processing_time:.4f} seconds")
    print(f"Throughput: {throughput_lps:.2f} lines/second")
    print(f"Throughput: {throughput_wps:.2f} words/second")
    print(f"Throughput: {throughput_wpms:.2f} words/millisecond")
    print("\n--- Latency (ms) ---")
    print(f"P50 (Median): {p50_latency:.4f} ms")
    print(f"P90: {p90_latency:.4f} ms")
    print(f"P99: {p99_latency:.4f} ms")
    print("-" * 28)


except FileNotFoundError:
    print(f"Error: The file '{input_file}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
