import tqdm
from torchvision.utils import make_grid


def Training(epochs,initial_epoch,train_dataloader,optimizer,model,loss_fn,save_fn,save_dir,devices,Tensorboard):
    tb_step=1
    for epoch in range(initial_epoch,epochs+1):
        
        prograss_bar=tqdm.tqdm(range(len(train_dataloader)),"Train proccess")
        
        for batch,img in enumerate(train_dataloader):
            
            img=img.to(devices)
            
            optimizer.zero_grad()
            out,loss_quantize=model(img)
            reconstruction_loss=loss_fn(out,img)
            loss = reconstruction_loss+loss_quantize
            
            loss.backward()
            optimizer.step()
            prograss_bar.update(1)   
            
            if batch==50:
                img_grid=make_grid(out,10)
                prograss_bar.set_postfix({"EPOCH":epoch,"step":batch+1,"LOSS": (loss.item()/(batch+1))})
                Tensorboard.add_scalar("Loss VQ-VAE",(loss.item()/(batch+1)),tb_step)
                Tensorboard.add_image("Pred img",img_grid,tb_step)
                tb_step+=1
        
        prograss_bar.close()
        save_fn(epoch,optimizer,model,save_dir)