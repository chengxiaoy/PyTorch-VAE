import math
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import joblib
from models import *
import yaml
from experiment import VAEXperiment
from tqdm import tqdm

# 5_o_Clock_Shadow：刚长出的双颊胡须Arched_Eyebrows：柳叶眉Attractive：吸引人的Bags_Under_Eyes：眼袋Bald：秃头Bangs：刘海
# Big_Lips：大嘴唇Big_Nose：大鼻子Black_Hair：黑发Blond_Hair：金发Blurry：模糊的Brown_Hair：棕发Bushy_Eyebrows：浓眉Chubby：圆胖的
# Double_Chin：双下巴Eyeglasses：眼镜Goatee：山羊胡子Gray_Hair：灰发或白发Heavy_Makeup：浓妆High_Cheekbones：高颧骨Male：男性
# Mouth_Slightly_Open：微微张开嘴巴Mustache：胡子，髭Narrow_Eyes：细长的眼睛No_Beard：无胡子Oval_Face：椭圆形的脸Pale_Skin：苍白的皮肤
# Pointy_Nose：尖鼻子 Receding_Hairline：发际线后移Rosy_Cheeks：红润的双颊Sideburns：连鬓胡子Smiling：微笑Straight_Hair：直发
# Wavy_Hair：卷发Wearing_Earrings：戴着耳环Wearing_Hat：戴着帽子Wearing_Lipstick：涂了唇膏Wearing_Necklace：戴着项链Wearing_Necktie：戴着领带Young：年轻人


attr_txt = '/data/kaggle/shared/Data/celeba/list_attr_celeba.txt'
with open(attr_txt) as f:
    content = f.readlines()
line = content[1].strip('\n')
attrs = line.split(' ')

SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.CenterCrop(148),
                                transforms.Resize(64),
                                transforms.ToTensor(),
                                SetRange])

celeba_test = CelebA(root="../shared/Data/", split="test", transform=transform, download=False)
celeba_train = CelebA(root="../shared/Data/", split="train", transform=transform, download=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_attr_dataloader(attr):
    label_index = attrs.index(attr)
    dataset_attr_index = []
    dataset_no_attr_index = []
    for index, (img, labels) in enumerate(celeba_test):
        if labels[label_index] == 1:
            dataset_attr_index.append(index)
        else:
            dataset_no_attr_index.append(index)

    attr_dataloader = DataLoader(celeba_test, batch_size=64, sampler=SubsetRandomSampler(dataset_attr_index))
    no_attr_dataloader = DataLoader(celeba_test, batch_size=64, sampler=SubsetRandomSampler(dataset_no_attr_index))

    return attr_dataloader, no_attr_dataloader


model_config = {"vae": "configs/vae.yaml"}
model_checkpoint_path = {"vae": '/data/kaggle/PyTorch-VAE/logs/VanillaVAE/version_16/checkpoints/_ckpt_epoch_25.ckpt'}


def get_model(model_name):
    config_file = model_config[model_name]
    path = model_checkpoint_path[model_name]
    with open(config_file, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    model = vae_models[config['model_params']['name']](**config['model_params'])
    model.to(device)

    experiment = VAEXperiment(model,
                              config['exp_params'])

    experiment.load_state_dict(torch.load(path)['state_dict'])
    return experiment.model


def get_attr_vector(model, attr_name):
    attr_dataloader, no_attr_dataloader = get_attr_dataloader(attr_name)

    attr_vectors = []
    for imgs, labels in tqdm(attr_dataloader):
        imgs = imgs.to(device)
        vectors = model.inference(imgs).detach().cpu().numpy()
        attr_vectors.append(np.mean(vectors, axis=0))

    no_attr_vectors = []
    for imgs, labels in tqdm(no_attr_dataloader):
        imgs = imgs.to(device)
        vectors = model.inference(imgs).detach().cpu().numpy()
        no_attr_vectors.append(np.mean(vectors, axis=0))
    attr_vect = np.mean(attr_vectors, axis=0) - np.mean(no_attr_vectors, axis=0)
    joblib.dump(attr_vect, model_name + "_" + attr_name + ".pkl")
    return attr_vect


attr_name = "5_o_Clock_Shadow"
model_name = "vae"
model = get_model(model_name)

attr_vect = get_attr_vector(model, attr_name)




