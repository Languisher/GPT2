# GPT2-Docker

Last update: Sep 21, 2024

Implementation of [GPT-2](https://huggingface.co/openai-community/gpt2) source code, however different from the original implementation, different components are separated into isolated parts.

## How to use this code

1. Download GPT2-Model to the local directory.
2. Execute main.py


## Misc.

### How to download GPT2-Model to local directory

> Thanks JunHan Liu for providing the source code.

Execute the bash code listed below:

```bash
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com

mkdir gpt2-model
huggingface-cli download --resume-download openai-community/gpt2 --local-dir gpt2-model
```

### Equivalent Functionality

The code is encapsulated in the official edition. Thanks to GPT-4o, the equivalent functionality of the code is as:

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load model
model_dir = "./components/gpt2-model"
model = GPT2LMHeadModel.from_pretrained(model_dir)
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)

# Inference Example
def inference(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    logging.info(f'input ids is: {input_ids}')
    
    # Generate text
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    
    logging.info(f'output is: {output}')
    
    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text

    
if __name__ == "__main__":
    prompt = "Once upon a time,"
    generated_text = inference(prompt)

    logging.info(f'Generated text is: {generated_text}')
```