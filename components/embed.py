from transformers import GPT2LMHeadModel
import torch
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d] -  %(message)s')

model_dir = "./components/gpt2-model"
model = GPT2LMHeadModel.from_pretrained(model_dir)

    

class EmbeddingHandler:
    def __init__(self,
                 ):
        self.wte = model.transformer.wte
        self.wpe = model.transformer.wpe

    def generate_position_ids(self, input_ids):
        return torch.arange(0, input_ids.size(-1)).unsqueeze(0).to(input_ids.device)

    def inference(self,
                  input_ids: torch.Tensor,
                  position_ids: torch.Tensor = None):
        # generate input token embed
        token_embeddings = self.wte(input_ids)

        # generate input token pos embed
        if position_ids is None:
            position_ids = self.generate_position_ids(input_ids)

        position_embeddings = self.wpe(position_ids)

        hidden_states = token_embeddings + position_embeddings

        logging.debug(f"hidden_states.shape: (batch, seq_length, embed_dim) {hidden_states.shape}")
        return hidden_states