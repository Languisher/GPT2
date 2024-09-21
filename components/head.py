from transformers import GPT2LMHeadModel
import torch
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d] -  %(message)s')

# Load GPT-2 model
model_dir = "./components/gpt2-model"
model = GPT2LMHeadModel.from_pretrained(model_dir)

class LMHeadHandler:
    def __init__(self):
        self.ln_f  = model.transformer.ln_f
        self.lm_head = model.lm_head

    def inference(self, hidden_states: torch.Tensor):
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        
        logging.debug(f"logits.shape: (batch, seq_length, vocab_dim) {logits.shape}")
        
        return logits
