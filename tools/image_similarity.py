from pathlib import Path
import sys
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from conch.open_clip_custom import create_model_from_pretrained
from torchvision import transforms as T
from transformers import AutoImageProcessor, AutoModel
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from collections import defaultdict
import numpy as np
from tqdm import tqdm

root = Path(sys.argv[1])

device = torch.device("cuda:0")

model_uni = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
tf_uni = create_transform(**resolve_data_config(model_uni.pretrained_cfg, model=model_uni))
model_uni = model_uni.to(device).eval()

model_conch, tf_conch = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch")
model_conch = model_conch.to(device).eval()


tf_phicon = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
model_phicon = AutoModel.from_pretrained("owkin/phikon-v2")
model_phicon.eval().to(device);

model_gigapath = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)

model_gigapath = model_gigapath.to(device).eval()

tf_gigapath = T.Compose(
    [
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

tf_fns = {
    "uni": tf_uni,
    "phiconv2": lambda x: tf_phicon(x).pixel_values[0],
    "gigapath": tf_gigapath,
    "conch": tf_conch,
}

enc_forward_fns = {
    "uni": lambda x: model_uni(x),
    "phiconv2": lambda x: model_phicon(pixel_values= x ).last_hidden_state[:,0,:],
    "gigapath": lambda x: model_gigapath(x).squeeze(),
    "conch": lambda x: model_conch.encode_image(x, proj_contrast=False, normalize=False),
}


class MyDataset(Dataset):
    def __init__(self, root, names, tf_fns):
        self.root = root
        self.names = names
        self.tf_fns = tf_fns

    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        out = {}

        for n in ["real", "syn"]:
            img_path = self.root / f"{self.names[idx]}_{n}.jpg"
            img = Image.open(img_path).convert("RGB")
            for k, tf in self.tf_fns.items():
                img_tf = tf(img)
                out[f"{n}_{k}"] = img_tf

        return out


img_list = list(root.iterdir())
img_list = set([item.stem for item in img_list])
names = list(set([item.split("_")[0] for item in img_list]))
names = [n for n in names if f"{n}_real" in img_list and f"{n}_syn" in img_list]

ds = MyDataset(root, names, tf_fns)
dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4)

similarities = defaultdict(list)

for batch in tqdm(dl):

    for enc, fw in enc_forward_fns.items():

        with torch.no_grad():
            try:
                rr = batch[f"real_{enc}"].to(device)
            except:
                breakpoint()
            ss = batch[f"syn_{enc}"].to(device)
            real_fw = fw(rr)
            syn_fw = fw(ss)

        sim = torch.nn.functional.cosine_similarity(real_fw, syn_fw)
        sim = sim.cpu().numpy().tolist()
        similarities[enc].extend(sim)


for enc, sim_lis in similarities.items():
    sim = np.array(sim_lis)
    m = np.mean(sim)

    print(f"Mean similarity for {enc}: {m:.4f}")



