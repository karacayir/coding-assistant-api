# AI Code Assistant Inference

## Overview

The **coding-assistant-api** repository serves as a Language Model (LLM) inference endpoint for a coding assistant. This project is designed to provide intelligent code suggestions and completions based on natural language input.

## Features

- **Language Model Inference:** Utilizes a powerful language model to understand and generate code suggestions.
- **Fast API Endpoint:** Provides a simple and scalable API for integrating the code assistant into various applications.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/karacayir/coding-assistant-api.git
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. Start the server:

    ```bash
    python app.py
    ```

2. Make a POST request to the API endpoint with natural language input:

    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"prompt": "Write a function that calculates the factorial of a number."}' http://localhost:5000/v1/completions
    ```

3. Receive intelligent code suggestions as a response.

## Configuration

The behavior of the code assistant can be customized by modifying the configuration file (`config.py`). Adjust parameters such as model type, confidence thresholds, and other settings to fine-tune the code suggestions.

## Contributing

We welcome contributions! If you have any ideas, improvements, or bug fixes, please open an issue or submit a pull request.

## Acknowledgments

- The underlying language model is powered by Phind-CodeLlama-34B-v2 which is a finetuned CodeLlama model.