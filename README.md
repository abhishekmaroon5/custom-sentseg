# Custom Sentence Segmentation

This repository contains a customizable sentence segmentation codebase tailored for specific use cases. Traditional sentence segmentation tools often fail on domain-specific texts or noisy data. This project provides a framework to build and train your own sentence segmentation model.

## Features

*   **Customizable:** Easily adapt the model to your specific domain and data.
*   **Trainable:** Train your own sentence segmentation models.
*   **Extensible:** The modular design allows for easy extension and experimentation with different models and techniques.
*   **Two Models:** Includes implementations for both a BERT-based model and a BiLSTM-CRF model.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/custom-sentseg.git
    cd custom-sentseg
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Preparation

The first step is to download and prepare the training data. The `billsum` dataset is used for this project.

```bash
python notebooks/01_download_data.py
```

This script will download the dataset, process it, and create a `train.txt` file in the `data/` directory.

### Training

This project includes two different models for sentence segmentation.

#### BiLSTM-CRF Model

To train the BiLSTM-CRF model, run the following command:

```bash
python train_crf.py
```

This will use the configuration from `configs/config_crf.yaml`, train the model, and save it to `models/bilstm_crf_model.pt`. The vocabulary will be saved to `models/word_to_idx.pt`.

#### BERT-based Model

To train the BERT-based model, run the following command:

```bash
python train_bert.py
```

This will use the configuration from `configs/config.yaml` and save the trained model to `models/sentence_segmentation_model.pt`.

### Inference

To perform sentence segmentation on a text, use the `segment.py` script. This script is designed for the BiLSTM-CRF model.

```bash
python segment.py --model_path models/bilstm_crf_model.pt --vocab_path models/word_to_idx.pt --text "This is the first sentence. This is the second sentence."
```

## Configuration

The project is configured using YAML files in the `configs/` directory:
*   `config.yaml`: Configuration for the BERT-based model.
*   `config_crf.yaml`: Configuration for the BiLSTM-CRF model.

This allows you to specify model architecture, training parameters, and data paths.

## Testing

This project includes scripts to test the accuracy and performance of the sentence segmentation.

### Accuracy Test

To run the accuracy test, you can use the `accuracy_test.py` script. This script uses the `spacy` library to compare the segmentation results.

*Note: The script requires the `en_core_web_sm` model from spaCy. If you don't have it installed, you can download it by running:*
```bash
python -m spacy download en_core_web_sm
```

*If you are using a conda environment, you can run:*
```bash
conda run -n your_env_name python -m spacy download en_core_web_sm
```

```bash
python accuracy_test.py
```

### Performance Test

To run the performance test, you can use the `performance_test.py` script.

```bash
python performance_test.py
```

## Contributing

Contributions are welcome! Please create an issue to discuss any changes you wish to make.

## License

This project is licensed under the MIT License. 