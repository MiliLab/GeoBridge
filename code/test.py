# -*- coding: utf-8 -*-
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import os
import yaml
from tqdm import tqdm
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
import random
import csv
from transformers import CLIPTokenizer
from math import radians, sin, cos, sqrt, atan2
random.seed(42)
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--test_dir', default='', type=str, help='./test_data')
parser.add_argument('--checkpoint_dir', default='', type=str, help='save checkpoints path')
parser.add_argument('--checkpoint', default='', type=str, help='save model path')
parser.add_argument('--use_fliplr', action='store_true', default=True, help='use horizontal flip in testing')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
opt = parser.parse_args()
config_path =os.path.join(opt.checkpoint_dir, 'opts.yaml')
with open(config_path, 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)
for cfg, value in config.items():
    if cfg == "gpu_ids":
        continue
    if cfg == "batchsize":
        continue
    setattr(opt, cfg, value)

def compute_L_metrics_top1(sim_matrix, q_sources, g_sources, thresholds_km=(0.05, 0.10, 0.15)):
    sim = sim_matrix.detach().cpu()
    top1_idx = torch.argmax(sim, dim=1)  
    n = sim.size(0)
    l_hits = [0, 0, 0]
    total_dev = 0.0
    valid = 0
    for i in range(n):
        try:
            q_lat, q_lon = extract_center_latlon_from_name(q_sources[i])
            g_lat, g_lon = extract_center_latlon_from_name(g_sources[int(top1_idx[i])])
        except Exception:
            continue
        d = haversine_km(q_lat, q_lon, g_lat, g_lon)
        total_dev += d
        valid += 1
        if d <= thresholds_km[0]:
            l_hits[0] += 1
        if d <= thresholds_km[1]:
            l_hits[1] += 1
        if d <= thresholds_km[2]:
            l_hits[2] += 1
    if valid == 0:
        return {"Count": 0, "Deviation": 0.0, "L@50": 0.0, "L@100": 0.0, "L@150": 0.0}
    return {
        "Count": valid,
        "Deviation": total_dev / valid,
        "L@50":  100.0 * l_hits[0] / valid,
        "L@100": 100.0 * l_hits[1] / valid,
        "L@150": 100.0 * l_hits[2] / valid
    }
def dump_topk_csv(sim_matrix, q_paths, g_paths, q_ids=None, g_ids=None, k=5, out_csv_path=None):
    sim = sim_matrix.detach().cpu()
    Nq, Ng = sim.shape
    kk = min(k, Ng)
    topk = torch.topk(sim, k=kk, dim=1, largest=True)
    idx = topk.indices         
    val = topk.values           
    if out_csv_path is None:
        return
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    with open(out_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_idx","query_id","query_path",
                    "rank","gallery_idx","gallery_id","gallery_path",
                    "score","correct"])
        for i in range(Nq):
            qid = q_ids[i] if q_ids is not None else ""
            for r in range(kk):
                j = int(idx[i, r])
                gid = g_ids[j] if g_ids is not None else ""
                score = float(val[i, r])
                correct = int(qid == gid) if (q_ids is not None and g_ids is not None) else ""
                w.writerow([i, qid, q_paths[i],
                            r+1, j, gid, g_paths[j],
                            score, correct])

class TextCSVDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        if 'text' not in df.columns:
            raise ValueError("No")
        self.texts = df['text'].astype(str).tolist()
        if 'id' in df.columns:
            self.ids = df['id'].astype(str).tolist()
        else:
            first_col = [c for c in df.columns if c != 'text'][0]
            self.ids = df[first_col].astype(str).tolist()
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.ids[idx]


def extract_text_features_hf(txt_loader, text_encoder, tokenizer, device):
    feats_list, ids_list, txt_list = [], [], []
    text_encoder.eval()
    with torch.no_grad():
        for texts, ids in tqdm(txt_loader):
            tok = tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            tok = {k: v.to(device, non_blocking=True) for k, v in tok.items()}
            try:
                feats = text_encoder(**tok)           
            except TypeError:
                feats = text_encoder(
                    tok["input_ids"],
                    tok.get("attention_mask", None)
                )
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            feats = F.normalize(feats, dim=1).cpu()

            feats_list.append(feats)
            ids_list.extend(ids)
            txt_list.extend(texts)

    T = torch.cat(feats_list, dim=0) if len(feats_list) > 0 else torch.empty(0)
    return T, ids_list, txt_list
def fliplr(img): 
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1, -1, -1).long().cuda() 
    img_flip = img.index_select(3, inv_idx)
    return img_flip
def get_sample_paths(dataset):
    if hasattr(dataset, "samples"):
        s = dataset.samples
        if len(s) == 0:
            return []
        first = s[0]
        if isinstance(first, (tuple, list)) and len(first) >= 1 and isinstance(first[0], str):
            return [x[0] for x in s]
        if isinstance(first, str):
            root = getattr(dataset, "root", None) or getattr(dataset, "root_dir", None)
            if root is not None:
                return [os.path.join(root, fname) if not os.path.isabs(fname) else fname for fname in s]
            else:
                return list(s)
    if hasattr(dataset, "imgs"):
        return [p for p, _ in dataset.imgs]
    raise AttributeError("No")

def extract_features_imagelevel(dataloader, encoder, device):
    feats_list, lbl_list, ids_list = [], [], []
    paths = get_sample_paths(dataloader.dataset)

    encoder.eval()
    with torch.no_grad():
        idx0 = 0
        for imgs, ids in tqdm(dataloader):
            imgs = imgs.to(device, non_blocking=True)
            feats = encoder(imgs)
            if opt.use_fliplr:
                feats = (feats + encoder(fliplr(imgs))) / 2.0
            feats = F.normalize(feats, dim=1)
            feats_list.append(feats.cpu())
            if torch.is_tensor(ids):
                lbl_list.append(ids.cpu())
                bs = imgs.size(0)
                batch_paths = paths[idx0: idx0 + bs]
                ids_list.extend([os.path.splitext(os.path.basename(p))[0] for p in batch_paths])
                idx0 += bs
            else:
                lbl_list.append(torch.full((len(ids),), -1, dtype=torch.long))  
                ids_list.extend(list(ids))
    feats = torch.cat(feats_list, dim=0)
    labels = torch.cat(lbl_list, dim=0)
    return feats, labels, paths, ids_list

def compute_similarity_matrix(Q, G, logit_scale=1.0, device='cuda', q_chunk=1024, g_chunk=4096):
    Nq, Ng = Q.size(0), G.size(0)
    S = torch.empty((Nq, Ng), dtype=torch.float32)
    for qs in range(0, Nq, q_chunk):
        qe = min(qs + q_chunk, Nq)
        q = Q[qs:qe].to(device)               
        row = torch.empty((qe - qs, Ng), dtype=torch.float32)
        for gs in range(0, Ng, g_chunk):
            ge = min(gs + g_chunk, Ng)
            g = G[gs:ge].to(device)          
            block = torch.matmul(q, g.t()) * logit_scale
            row[:, gs:ge] = block.cpu()
        S[qs:qe] = row
    return S

def topk_by_id(sim_matrix, q_ids, g_ids, topk=(1,5)):
    sim = sim_matrix.detach().cpu()
    max_k = max(topk)
    idx = torch.topk(sim, k=max_k, dim=1, largest=True).indices
    correct = {k: 0 for k in topk}
    for i in range(idx.size(0)):
        cand = [g_ids[j] for j in idx[i].tolist()]
        for k in topk:
            if q_ids[i] in cand[:k]:
                correct[k] += 1
    n = len(q_ids)
    return {k: correct[k]/n for k in topk}

def test_text_to_image_imagelevel(model, opt,txt_loader,gallery_loader,text_encoder,gallery_encoder,tokenizer,device="cuda",k_dump=5,out_name="top5_text2img_byid.csv"):
    base = model.module if isinstance(model, torch.nn.DataParallel) else model
    logit_scale = base.model_logit_scale.exp().detach().item() if hasattr(base, "model_logit_scale") else 1.0
    print("Extracting text features...")
    T, t_ids, t_texts = extract_text_features_hf(txt_loader, text_encoder, tokenizer, device)
    print("Extracting gallery (image-level) features...")
    G, g_labels, g_paths, g_ids = extract_features_imagelevel(gallery_loader, gallery_encoder, device)
    S = compute_similarity_matrix(
        T, G, logit_scale=logit_scale, device=device,
        q_chunk=getattr(opt, "score_chunk", 1024),
        g_chunk=max(2048, getattr(opt, "score_chunk", 1024)*4)
    )
    topk = topk_by_id(S, t_ids, g_ids, topk=(1,5))
    mAP, r1p = compute_map_and_recall_id(S, t_ids, g_ids, top_percent=0.01)
    print(f"[Text→Image | by file id] Top-1: {topk[1]:.6f}  Top-5: {topk[5]:.6f}  mAP: {mAP:.6f}  R@1%: {r1p:.6f}")
    out_csv = os.path.join(opt.checkpoint_dir, out_name)
    dump_topk_csv(S, q_paths=t_texts, g_paths=g_paths, q_ids=t_ids, g_ids=g_ids, k=k_dump, out_csv_path=out_csv)
    print(f"[Save] 文本→图像 Top-{k_dump} 已保存: {out_csv}")
    try:
        L_txtimg = compute_L_metrics_top1(
                    S,
                    q_sources=t_ids,      
                    g_sources=g_ids,      
                    thresholds_km=(0.05, 0.10, 0.15)
                )
        print("\n[Geo L@ | Text→Image | Top-1]")
        print(f"Count: {L_txtimg['Count']}")
        print(f"Avg Deviation: {L_txtimg['Deviation']:.4f} km")
        print(f"L@50 : {L_txtimg['L@50']:.2f}%")
        print(f"L@100: {L_txtimg['L@100']:.2f}%")
        print(f"L@150: {L_txtimg['L@150']:.2f}%")
        out_metrics2 = os.path.join(opt.checkpoint_dir, "geo_L_text2image.txt")
        with open(out_metrics2, "w") as f:
            f.write("[Geo L@ | Text→Image | Top-1]\n")
            for k in ["Count","Deviation","L@50","L@100","L@150"]:
                f.write(f"{k}: {L_txtimg[k] if k=='Count' else f'{L_txtimg[k]:.6f}'}\n")
        print(f"[Save] L@ 指标已保存: {out_metrics2}\n")
    except Exception as e:
        print(f"[Warn] 计算 L@(Text→Image) 失败：{e}")

def test_model_imagelevel(model, opt, query_loader, gallery_loader, query_encoder, gallery_encoder,
                           device="cuda"):
    model.eval()
    print("Extracting query (image-level) features...")
    Q, q_labels, q_paths, q_ids = extract_features_imagelevel(query_loader,  query_encoder,  device)
    print("Extracting gallery (image-level) features...")
    G, g_labels, g_paths, g_ids = extract_features_imagelevel(gallery_loader, gallery_encoder, device)
    base = model.module if isinstance(model, torch.nn.DataParallel) else model
    logit_scale = base.model_logit_scale.exp().detach().item() if hasattr(base, "model_logit_scale") else 1.0
    S = compute_similarity_matrix(Q, G, logit_scale=logit_scale,
                                  device=device,
                                  q_chunk=getattr(opt, "score_chunk", 1024),
                                  g_chunk=max(2048, getattr(opt, "score_chunk", 1024)*4))
    out_csv = os.path.join(opt.checkpoint_dir, "top5_t2d_byid.csv") 
    dump_topk_csv(S, q_paths, g_paths, q_ids=q_ids, g_ids=g_ids, k=5, out_csv_path=out_csv)
    print(f"[Save] 已保存前5匹配到: {out_csv}")
    topk = topk_by_id(S, q_ids, g_ids, topk=(1,5))
    mAP, r1p = compute_map_and_recall_id(S, q_ids, g_ids, top_percent=0.01)
    print(
        f"[Image-Level | by file id] "
        f"Top-1: {topk[1]:.6f}  Top-5: {topk[5]:.6f}  "
        f"mAP: {mAP:.6f}  R@1%: {r1p:.6f}"
    )
def compute_map_and_recall_id(similarity_matrix, query_ids, gallery_ids, top_percent=0.01):
    sim = similarity_matrix.detach().cpu()
    Nq, Ng = sim.shape
    k_top = max(1, int(Ng * top_percent))
    APs = []
    recall_hits = 0
    for i in range(Nq):
        target = query_ids[i]
        order = torch.argsort(sim[i], descending=True).tolist()
        ranked_ids = [gallery_ids[j] for j in order]
        try:
            pos = ranked_ids.index(target)  
        except ValueError:
            continue
        APs.append(1.0 / float(pos + 1))
        if pos < k_top:
            recall_hits += 1
    mAP = float(np.mean(APs)) if APs else 0.0
    recall_top1p = recall_hits / float(Nq) if Nq > 0 else 0.0
    return mAP, recall_top1p

data_transforms = transforms.Compose([
    transforms.Resize((opt.h, opt.w), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
class ImageFolderFlat(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"路径不存在: {root_dir}")
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        self.samples = [f for f in os.listdir(root_dir) if f.lower().endswith(exts)]
        self.samples.sort()

        if len(self.samples) == 0:
            raise RuntimeError(f"{root_dir} 下没有图像文件")
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename = self.samples[idx]
        path = os.path.join(self.root_dir, filename)
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        file_id = os.path.splitext(filename)[0]
        return img, file_id

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def extract_center_latlon_from_name(s: str):
    base = os.path.splitext(os.path.basename(str(s)))[0]
    m = _LON_LAT_4F.search(base) or _LON_LAT_4F.search(str(s))
    if not m:
        raise ValueError(f"无法从 {s} 中解析 lon_lat_lon_lat")
    lon1, lat1, lon2, lat2 = map(float, m.groups())
    lon_c = (lon1 + lon2) / 2.0
    lat_c = (lat1 + lat2) / 2.0
    return lat_c, lon_c  

if __name__ == '__main__':
    print(opt.gpu_ids)
    str_ids = opt.gpu_ids.split(',')
    gpu_ids = [int(x) for x in str_ids if int(x) >= 0]
    if torch.cuda.is_available() and len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        device = torch.device(f"cuda:{gpu_ids[0]}")
    else:
        device = torch.device("cpu")
    print(os.path.join(opt.checkpoint_dir, opt.checkpoint))
    model = torch.load(os.path.join(opt.checkpoint_dir, opt.checkpoint),weights_only=False, map_location="cpu")
    model = model.to(device)
    model.eval()

    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    drone_encoder = base_model.drone_encoder
    satellite_encoder = base_model.satellite_encoder
    street_encoder = base_model.street_encoder
    text_encoder = base_model.text_encoder
    if torch.cuda.is_available() and len(gpu_ids) > 1:
        drone_encoder = torch.nn.DataParallel(drone_encoder, device_ids=gpu_ids)
        satellite_encoder = torch.nn.DataParallel(satellite_encoder, device_ids=gpu_ids)
        street_encoder = torch.nn.DataParallel(street_encoder, device_ids=gpu_ids)
        text_encoder = torch.nn.DataParallel(text_encoder, device_ids=gpu_ids)
    drone_encoder = drone_encoder.to(device)
    satellite_encoder = satellite_encoder.to(device)
    street_encoder = street_encoder.to(device)
    text_encoder = text_encoder.to(device)
    query_set = ImageFolderFlat(os.path.join(opt.test_dir, 'satellite'), transform=data_transforms)
    gallery_set = ImageFolderFlat(os.path.join(opt.test_dir, 'street'), transform=data_transforms)
    text_set = TextCSVDataset(os.path.join(opt.test_dir, 'drone.csv'))
    query_loader   = DataLoader(query_set,   batch_size=opt.batchsize, shuffle=False, num_workers=opt.num_worker)
    gallery_loader = DataLoader(gallery_set, batch_size=opt.batchsize, shuffle=False, num_workers=opt.num_worker)
    txt_loader = DataLoader(text_set, batch_size=opt.batchsize, shuffle=False, num_workers=getattr(opt, "num_worker", 4),
                            collate_fn=lambda batch: ( [x[0] for x in batch], [x[1] for x in batch] ))
    print(len(gpu_ids))
    print(opt.batchsize)
    test_model_imagelevel(base_model, opt,query_loader, gallery_loader,query_encoder=satellite_encoder,   gallery_encoder=street_encoder,device=device)
    TEXT_TOKENIZER_PATH = "clip-vit-large-patch14"
    tokenizer = CLIPTokenizer.from_pretrained(TEXT_TOKENIZER_PATH)
    test_text_to_image_imagelevel(model=base_model,opt=opt,txt_loader=txt_loader,gallery_loader=query_loader,text_encoder=base_model.text_encoder,
                                    gallery_encoder=satellite_encoder,tokenizer=tokenizer, k_dump=5, out_name="top5_text2drone_byid.csv")