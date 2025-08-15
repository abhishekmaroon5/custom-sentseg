
import os
from datasets import load_dataset

def download_and_prepare_dataset(dataset_name, data_dir):
    """
    Downloads and prepares a dataset for sentence segmentation.

    Args:
        dataset_name (str): The name of the dataset to download from the Hugging Face Hub.
        data_dir (str): The directory to save the dataset to.
    """
    # Create the data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Load the dataset
    print(f"Downloading dataset: {dataset_name}...")
    dataset = load_dataset(dataset_name, split='train')
    print("Dataset downloaded successfully.")

    # For this example, we'll just use a subset of the data
    # to speed things up.
    subset = dataset.select(range(5000))

    # We will treat each article as a document and save it to a text file.
    # The task will be to segment the text into sentences.
    # We will create a single text file where each line is a sentence.
    # We will add a special token "<eos>" at the end of each sentence
    # and join the sentences to form a long text.
    # We will save this text to a file in the data directory.

    output_file = os.path.join(data_dir, "train.txt")
    print(f"Processing and saving data to {output_file}...")

    with open(output_file, "w", encoding="utf-8") as f:
        for example in subset:
            # We are assuming that the dataset provides text that is already
            # segmented into sentences. If not, this part needs to be adjusted.
            # Here, we'll use the 'summary' feature of the billsum dataset.
            sentences = example['summary'].strip().split('\n')
            for sentence in sentences:
                if sentence:
                    f.write(sentence.strip() + " <eos>\n")
    
    print("Data processing complete.")

if __name__ == "__main__":
    DATASET_NAME = "billsum"
    DATA_DIR = "data"
    download_and_prepare_dataset(DATASET_NAME, DATA_DIR) 