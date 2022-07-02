from pathlib import Path
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from torch.utils.data import DataLoader
import cv2

"""
This is a very fast scratch for building a gender classification model for my data preprocessing tasks..
Accuracy on Validation (10K Images): 97.6%
"""
class GenderClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.setup()

    def setup(self):
        self.backbone = nn.Sequential(*[m for m in resnet18(pretrained=True).children() if not isinstance(m, nn.modules.linear.Linear)])
        self.fc1 = nn.Linear(512, 1)
        self.act1 = nn.Sigmoid()
        self.bce = nn.BCELoss()
        self.otim = torch.optim.Adam(list(self.parameters()), lr=self.cfg['lr'], betas=(0, 0.999))
    
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.act1(x)

        return x

class GenderDataset(Dataset):
    def __init__(self, img_dir, resize):
        self.img_dir = img_dir
        self.transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), Resize([resize,resize])])
        self.preprocess()

    def preprocess(self):
        if self.img_dir is not None:
            self.all_gender = list(Path(self.img_dir).glob("**/*.jpg"))
      
    def __len__(self):
        return len(self.all_gender)

    def pad_image(self, img):
        pass
      
    def __getitem__(self, idx):
        img_path = self.all_gender[idx] 
        img = cv2.imread(img_path.as_posix())
        img = self.transform(img)
        gender = img_path.parent.as_posix().split("/")[-1]
        if gender == "male":
          g = 0
        else:
          g = 1
        return img, g

class GenderInferenceDataset(GenderDataset):

    def __getitem__(self, idx):
        img_path = self.all_gender[idx] 
        img = cv2.imread(img_path.as_posix())
        img = self.transform(img)
        return img, img_path.stem
     
def train(cfg):
    train_dataset = GenderDataset(img_dir=cfg['train_img_dir'], resize=cfg['resize'])
    val_dataset = GenderDataset(img_dir=cfg['val_img_dir'], resize=cfg['resize'])
    train_data_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])
    val_data_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])
    train_steps, val_steps = len(train_data_loader), len(val_data_loader)
    model = GenderClassifier(cfg)
    model.cuda()
    for i in range(cfg['max_epoches']):
        train_data_iter = iter(train_data_loader)
        val_data_iter = iter(val_data_loader)
        for j in range(train_steps):
            train_img, train_label = train_data_iter.next()
            model.train()
            result = model(train_img.cuda())
            loss = model.bce(result, train_label.cuda().unsqueeze(-1).to(result.dtype)).mean()
            if j % 100 == 0 :
                print(f" Epoch {i}: Training Steps: {j}/{train_steps} BCE Loss: {loss.item()}")
            model.otim.zero_grad()
            loss.backward()
            model.otim.step()
        if i % cfg['ckpt_epoch_interval'] == 0:
            print("Saving model ckpt..")
            torch.save(model.state_dict(), Path(cfg['ckpt_path'], f"Epoch{i}-model.ckpt"))
        if i % cfg['val_interval'] == 0:
            val_acc_total = 0
            for k in range(val_steps):
                val_img, val_label = val_data_iter.next()
                model.eval()
                with torch.no_grad():
                    result = model(val_img.cuda())
                    val_loss = model.bce(result, val_label.cuda().unsqueeze(-1).to(result.dtype)).mean()
                    pred = torch.where(result> 0.5, 1., 0.)
                    val_acc = torch.sum(pred == val_label.cuda().unsqueeze(-1)) / val_label.size(0)
                    val_acc_total += val_acc
                    if k % 100 == 0 :
                        print(f" Epoch {i}: Validation Steps: {k}/{val_steps} BCE Loss: {val_loss.item()} ACC: {val_acc}")
            print(f"Average Val Acc: {val_acc_total/val_steps}")

def inference(cfg):
    predictions = {}
    inference_dataset = GenderInferenceDataset(img_dir=cfg['inference_img_dir'], resize=cfg['resize'])
    inference_data_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False, num_workers=cfg['num_workers'])
    model = GenderClassifier(cfg)
    model.load_state_dict(torch.load(cfg['load_ckpt']))
    model.cuda()
    model.eval()
    inference_data_iter = iter(inference_data_loader)
    for i in range(len(inference_data_loader)):
        img, name = inference_data_iter.next()
        model.eval()
        result = model(img.cuda())
        pred = torch.where(result> 0.5, 1, 0)
        if pred[0].item() == 0:
            predictions[name[0]] = "Male"
        else:
            predictions[name[0]] = "Female"

    print(predictions)

    
if __name__ == "__main__":
    #TODO: No argparse was used for faster development.. Needa modified the below cfg..

    cfg = {"train_img_dir": "", 
            "val_img_dir": "",
            "ckpt_path": ".",
            "batch_size": 16, 
            "num_workers": 4, 
            "max_epoches":30, 
            "val_interval":3,
            "ckpt_epoch_interval":2,
            "lr":0.001,
            "resize":128,
            "mode":"inference",
            "inference_img_dir":"",
            "load_ckpt":""}

    if cfg['mode'] == "train":
        train(cfg)
    else:
        inference(cfg)  
