import torch
import torch.nn as nn
class Latent_Space(nn.Module):
    def __init__(self, num_embedding=128,hidden_dim=256):
        super(Latent_Space,self).__init__()
        
        self.embedding=nn.Embedding(num_embeddings=num_embedding,embedding_dim=hidden_dim)
        
    # Latent Space ,calculate distances between embeddings and encoder output:
    def latent_space(self,pre_quatized):
        
        # Permute data because last dimesion should be channel
        data_perm=pre_quatized.permute(0,3,2,1)
        
        # Reshape (batch_size,last_img_size*last_img_size,channel)
        quanted_in=data_perm.reshape(data_perm.shape[0],-1,data_perm.shape[-1])
        
        # Claculate distance
        dist=torch.cdist(quanted_in,self.embedding.weight[None,:].repeat(quanted_in.shape[0],1,1))
        
        # Min distance index
        min_distance_indexes=torch.argmin(dist,dim=-1)
        
        # Change
        quanted_out=torch.index_select(self.embedding.weight,0,min_distance_indexes.view(-1))
        
        return quanted_in,quanted_out
    

    def forward(self,pre_quantized):
        
        # Quatized vector with embeddings vector
        quanted_in,quanted_out=self.latent_space(pre_quantized)
        quanted_in=quanted_in.reshape((-1,quanted_in.size(-1)))
        
        # Calculate Quantize Loss
        loss1=torch.mean((quanted_out.detach()-quanted_in)**2) 
        loss2=torch.mean((quanted_out-quanted_in.detach())**2)  
        loss_quantize=loss1+(0.25*loss2)                        
        
        quanted_out=quanted_in+(quanted_out-quanted_in).detach()
        
        # Reshape for decoder input (batch_size,channel,last_img_size,last_img_isze)
        B,C,H,W=pre_quantized.shape 
        quanted_out_reshaped=quanted_out.reshape(B,H,W,C).permute(0,3,1,2)  # reshape to be decoder input that its shape İS like the encoder output
        
        return quanted_out_reshaped,loss_quantize





