from transformers import GPT2LMHeadModel
import torch
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d] -  %(message)s')

model_dir = "./components/gpt2-model"
model = GPT2LMHeadModel.from_pretrained(model_dir)

    
class TransformerLayerHandler:
    def __init__(self,
                 idx
                 ):
        self.idx = idx
        self.ln_1 = model.transformer.h[idx].ln_1
        self.attn = model.transformer.h[idx].attn
        self.ln_2 = model.transformer.h[idx].ln_2
        self.mlp = model.transformer.h[idx].mlp

    def inference(self,
                  hidden_states):
        hidden_states_ln_1 = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states_ln_1)[0]  # [0] extracts the output from the tuple
        hidden_states_attn = hidden_states + attn_output  # Residual connection
        
        hidden_states_ln_2 = self.ln_2(hidden_states_attn)
        mlp_output = self.mlp(hidden_states_ln_2)
        hidden_states_mlp = hidden_states_attn + mlp_output  # Residual connection

        return hidden_states_mlp

        

class BlockHandler:
    def __init__(self,
                 idx,
                 ):
        self.idx = idx
    
    def inference(self,
                  hidden_states):
        hidden_states = TransformerLayerHandler(self.idx).inference(hidden_states)
        
        logging.debug(f"Block {self.idx} - hidden_states.shape: (batch, seq_length, embed_dim) {hidden_states.shape}")
        return hidden_states