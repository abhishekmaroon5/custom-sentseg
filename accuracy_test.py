import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Define the input file path
input_file = "test_files/test_cases.txt"

print(f"--- Accuracy Test for Sentence Segmentation on {input_file} ---")

# Read the file and process each line
try:
    with open(input_file, 'r') as f:
        # Read lines and strip leading/trailing whitespace
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # Process each line from the file
    for line in lines:
        print(f"\\nOriginal Text: '{line}'")
        # Process the text with spaCy
        doc = nlp(line)
        
        print("Segmented Sentences:")
        i = 1
        for sent in doc.sents:
            print(f"  {i}. {sent.text}")
            i += 1
        print("-" * 20)

except FileNotFoundError:
    print(f"Error: The file '{input_file}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}") 