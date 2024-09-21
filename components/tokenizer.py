from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d] -  %(message)s')

model_dir = "./components/gpt2-model"
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)

class TokenizationHandler():
    def __init__(
        self,
    ):
        self.tokenizer = tokenizer
        # self.tokenizer.pad_token = self.tokenizer.eos_token
            

    def inference(
        self,
        input_txt: str,
    ):
        """
        Transform prompt_txt into tokenized input
        
        Params:
            input_txt is the prompt input text, string format
            
        Returns:
            A Dict variable, which consists of 'input_ids' and 'attention_masks'
        """

        hidden_states = self.tokenizer(
            input_txt,
            return_tensors='pt',  # Return PyTorch tensors
            truncation=True,      # Truncate sequences to the model's max length
            padding=False
            # padding='max_length'  # Pad sequences to the model's max length
        )

        input_ids = hidden_states["input_ids"]

        logging.debug(f"input_ids.shape: (batch, seq_length, embed_dim) {input_ids.shape}")
        return hidden_states


    def decode(self, input_ids, skip_special_tokens=True):
        """
        Decode token IDs back into a string.

        Params:
            input_ids (torch.Tensor or list): The token IDs to decode.
            skip_special_tokens (bool): Whether to skip special tokens like <eos>.

        Returns:
            str: The decoded text.
        """

        # If input_ids is a tensor, convert it to a list
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()

        # If input_ids is a list of lists (batch), decode each separately
        if isinstance(input_ids, list) and isinstance(input_ids[0], list):
            decoded_text = [self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens) for ids in input_ids]
            return decoded_text

        # If input_ids is a single list
        elif isinstance(input_ids, list):
            decoded_text = self.tokenizer.decode(input_ids, skip_special_tokens=skip_special_tokens)
            return decoded_text

        else:
            raise TypeError("input_ids must be a torch.Tensor or a list of token IDs.")