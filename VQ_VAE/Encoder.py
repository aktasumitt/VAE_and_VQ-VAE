import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, channel_size,hidden_dim=256):
        super(Encoder,self).__init__()
        
        self.hidden_dim=hidden_dim
        
        self.conv_block=nn.Sequential(nn.Conv2d(channel_size,hidden_dim,kernel_size=4,stride=2,padding=1,padding_mode="reflect"),
                                      nn.ReLU(),
                                      nn.Conv2d(hidden_dim,hidden_dim,kernel_size=4,stride=2,padding=1,padding_mode="reflect")
                                      )
        
        self.residual_block1=self.residual()
        self.residual_block2=self.residual()
    
    def residual(self):
        residual=nn.Sequential(nn.ReLU(),
                               nn.Conv2d(self.hidden_dim,self.hidden_dim,kernel_size=3,padding=1,padding_mode="reflect"),
                               nn.ReLU(),
                               nn.Conv2d(self.hidden_dim,self.hidden_dim,kernel_size=1))
        
        return residual

    def forward(self,x):
        
        x=self.conv_block(x)
        r1=self.residual_block1(x)+x
        out=self.residual_block2(r1)+r1
        return out
    
    
# model_enc=Encoder(3).to("cuda")
# x=torch.randn((2,3,128,128)).to("cuda")
# out=model_enc(x)




