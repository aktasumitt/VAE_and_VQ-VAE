import torch
import model,config,train,dataset,visualize,checkpoint
from torch.utils.tensorboard import SummaryWriter

# device
devices=("cuda" if torch.cuda.is_available() else "cpu")

# Tensorboard
Tensorboard=SummaryWriter("Tensorboard")

# Loading Dataset
full_img_list=dataset.Loading_Dataset(dataset_dir=config.DATASET_DIR)

# Create Dataset
Train_Dataset=dataset.Datasets(full_img_list=full_img_list)
print("Dataset Shape",Train_Dataset[-1].shape)

# Random Split
Train_Dataset,Test_Dataset=dataset.Random_split(dataset=Train_Dataset,test_split=config.TEST_SIZE)

# Create Dataloader
train_dataloader,test_dataloader=dataset.Dataloader(train=Train_Dataset,test=Test_Dataset,batch_size=config.BATCH_SIZE)

# Model
Model_Vae=model.VQ_VAE(config.CHANNEL_SIZE,hidden_dim=config.HIDDEN_DIM,num_embeddings=config.NUM_EMBEDDING).to(devices)

# Optimizer
optimizer=torch.optim.Adam(params=Model_Vae.parameters(),lr=config.LR)

# Reconsturacter Loss
loss_fn=torch.nn.MSELoss(reduction="sum")

# Loading Checkpoint if you have
STARTING_EPOCH=checkpoint.Load_Checkpoint(LOAD=config.LOAD_MODEL,checkpoint_dir=config.SAVE_PATH,model=Model_Vae,optimizer=optimizer)

# Training
train.Training(epochs=config.EPOCHS,initial_epoch=STARTING_EPOCH,train_dataloader=train_dataloader,optimizer=optimizer,model=Model_Vae,loss_fn=loss_fn,
               save_fn=checkpoint.Save_Checkpoint,save_dir=config.SAVE_PATH,devices=devices,Tensorboard=Tensorboard)

# Visualize
visualize.Visualize_Test(Test_dataloader=train_dataloader,model=Model_Vae,devices=devices)