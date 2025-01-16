import torch.nn as nn
from Decoder import Decoder
from Encoder import Encoder
from Embedding import Latent_Space

class VQ_VAE(nn.Module):
    def __init__(self,channel_size,hidden_dim=256,num_embeddings=128):
        super().__init__()
        
        self.encoder=Encoder(channel_size,hidden_dim) # Encoder Model
        
        self.latent_space=Latent_Space(num_embedding=num_embeddings,hidden_dim=hidden_dim) # Latent space (embedding)
        
        self.decoder=Decoder(channel_size,hidden_dim) # Decoder Model
        
    
    def forward(self,x):
        
        pre_quantized=self.encoder(x)
        
        quantize_out,loss_quantize=self.latent_space(pre_quantized)
        
        out_decoder=self.decoder(quantize_out)
        
        return out_decoder,loss_quantize

        
        
        
        
