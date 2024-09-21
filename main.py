import logging
import torch
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d] -  %(message)s')

from components.tokenizer import TokenizationHandler
from components.embed import EmbeddingHandler
from components.block import BlockHandler
from components.head import LMHeadHandler


class GenerateTextHandler:
    def __init__(self,
                 block_number: int = 12,
                 ):
        self._block_number = block_number # GPT2Block number
        self._tokenizer = TokenizationHandler()
        self._embedder = EmbeddingHandler()
        self._blocks = []

        for idx in range(self._block_number):
            block = BlockHandler(idx=idx)
            self._blocks.append(block)
        
        self._head = LMHeadHandler()

    def tokenize(self, 
                 prompt_text: str):
        return self._tokenizer.inference(prompt_text)

    def embedding(self,
                 input_ids):
        return self._embedder.inference(input_ids)
    
    def block(self,
              hidden_states,
              ):
        for block in self._blocks:
            hidden_states = block.inference(hidden_states)
        
        return hidden_states

    def head(self,
             hidden_states):
        return self._head.inference(hidden_states)

    def transform_logits(self,
                         logits):

        next_token_logits = logits[:, -1, :]  # Assuming batch size of 1
        
        # Convert logits to probabilities (optional, depending on your sampling method)
        probabilities = torch.softmax(next_token_logits, dim=-1)
        
        # Sample or select the next token
        next_token_id = torch.argmax(probabilities, dim=-1)  # Greedy decoding
        # Or use sampling:
        # next_token_id = torch.multinomial(probabilities, num_samples=1)

        # input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=-1)
        # generated_text = self._tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)

        next_token_id = next_token_id.unsqueeze(-1)
        
        logging.debug(f"next_token_id.shape: (batch, seq_length, vocab_dim) {next_token_id.shape}")
        return next_token_id
        
        

    def inference(
        self,
        prompt_text: str,
        max_text_length: int = 1,
    ):
        """
            All-in-one inference handler: execute all the steps, go through all of the components one by one.
        """
        logging.debug(f"input prompt_text: {prompt_text}")

        input_ids = self.tokenize(prompt_text)["input_ids"]

        for step in range(max_text_length):
            logging.debug(f"Generation step {step + 1}/{max_text_length}")

            hidden_states = self.embedding(input_ids)
            
            seq_length = input_ids.size(1)
            hidden_states = self.block(hidden_states)

            logits = self.head(hidden_states)
                
            next_token_id = self.transform_logits(logits)
            
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)  # Shape: [1, seq_len + 1]
            logging.debug(f"Updated input_ids shape: (batch, seq_length) {input_ids.shape}")
            
        logging.debug(f'output is: {input_ids}')
        
        generated_text = self._tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)
        return generated_text
        # return generated_text
    
        
if __name__ == "__main__":
    prompt_text = "Once upon a time,"
    print(GenerateTextHandler().inference(prompt_text, max_text_length=12))