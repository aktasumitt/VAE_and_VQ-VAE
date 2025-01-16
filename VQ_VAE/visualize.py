import torch
import matplotlib.pyplot as plt


def Visualize_Test(Test_dataloader,model,devices):
    with torch.no_grad():
        for batch,(img_test,_) in enumerate(Test_dataloader):
            
            img_test=img_test.to(devices)
            
            out_img,_=model(img_test)
            
            if batch==0:
                break
    
          
        
    for i in range(20):
        plt.subplot(4,5,i+1)
        out_img=out_img.cpu()
    
        plt.imshow(torch.permute(out_img[i],(1,2,0)))
        
    plt.show()