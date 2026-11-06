"""5-fold patient-level cross-validation on the 257 train/val bags (more positives than the 67-bag test set).
Per fold: train a fresh CEUS+B-mode teacher on the fold's training bags (no leakage), then each student variant.
Each fold model is also scored on the official held-out test set (5-model ensemble per variant)."""
from __future__ import annotations

import argparse, copy, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

from data.index import get_or_build_index
from data.splits import get_or_make_split
from data.transforms import IMAGENET_MEAN, IMAGENET_STD
from eval.metrics import classification_metrics
from experiments.cache import build_cache
from experiments.model import PlusNet
from experiments.train_variant import CachedDS, VARIANTS, W, predict, tf
from models.teacher import TeacherNet
from training.engine import EarlyStopper
from utils.config import load_config
from utils.seed import resolve_device, set_seed

OUT = Path("results/cv")
CV_VARIANTS = ["baseline", "feat", "clin", "feat_clin", "feat_clin_recon"]


class PairDS(Dataset):
    def __init__(self, recs, train):
        self.recs = [r for r in recs if r["ceus"] is not None]; self.t = tf(train)

    def __len__(self): return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        a = self.t(image=np.stack([r["img"]] * 3, -1), ceus=r["ceus"])
        c = self.t(image=np.stack([r["ceus"]] * 3, -1), ceus=r["ceus"])
        return dict(bmode=a["image"], ceus=c["image"], label=torch.tensor(r["label"]), bag_id=r["bag_id"])


def train_teacher(tr, va, device, epochs=10):
    t = TeacherNet("resnet18", pretrained=True, fusion="concat").to(device)
    opt = torch.optim.Adam(t.parameters(), lr=3e-4, weight_decay=1e-4)
    trl = DataLoader(PairDS(tr, True), batch_size=16, shuffle=True); val = DataLoader(PairDS(va, False), batch_size=16)
    best, best_auc = None, -1
    for ep in range(epochs):
        t.train()
        for b in trl:
            loss = F.cross_entropy(t(b["ceus"].to(device), b["bmode"].to(device)), b["label"].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        t.eval(); p, y = [], []
        with torch.no_grad():
            for b in val:
                p += F.softmax(t(b["ceus"].to(device), b["bmode"].to(device)), 1)[:, 1].cpu().tolist(); y += b["label"].tolist()
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y, p)
        if auc > best_auc: best_auc, best = auc, copy.deepcopy(t.state_dict())
    t.load_state_dict(best); t.eval()
    return t, best_auc


@torch.no_grad()
def make_targets(t, recs, device):
    tg = {}
    for b in DataLoader(PairDS(recs, False), batch_size=16):
        lg, ft = t(b["ceus"].to(device), b["bmode"].to(device), return_features=True)
        for i, l, f in zip(b["bag_id"], lg.cpu(), ft.cpu()): tg[i] = (l, f)
    return tg


def fit_student(variant, seed, tr, va, evals, tg, device, epochs=12, patience=5):
    v = VARIANTS[variant]; set_seed(seed)
    mk = lambda r, train: DataLoader(CachedDS(r, train), batch_size=16, shuffle=train)
    trl, val = mk(tr, True), mk(va, False)
    model = PlusNet(desc=bool(v.get("clin")), recon=bool(v.get("recon"))).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4); stop = EarlyStopper(patience, "max"); best = None
    for ep in range(epochs):
        model.train()
        for b in trl:
            x, y = b["image"].to(device), b["label"].to(device); o = model(x)
            loss = F.cross_entropy(o["logits"], y)
            if v.get("feat"):
                ft = torch.stack([tg[i][1] for i in b["bag_id"]]).to(device)
                loss = loss + W["feat"] * ((o["feats"] - ft) ** 2).mean()
            if v.get("clin"):
                loss = loss + v.get("w_clin", W["clin"]) * F.binary_cross_entropy_with_logits(o["desc"], b["desc"].to(device))
            if v.get("recon"):
                m = b["has_ceus"].to(device)
                if m.sum() > 0:
                    loss = loss + v.get("w_recon", W["recon"]) * ((o["ceus_hat"] - b["ceus"].to(device)).abs().mean((1, 2, 3)) * m).sum() / m.sum()
            opt.zero_grad(); loss.backward(); opt.step()
        _, vp, vy = predict(model, val, device)
        if stop.step(classification_metrics(vy, vp)["auc"]): best = copy.deepcopy(model.state_dict())
        if stop.should_stop: break
    model.load_state_dict(best)
    res = {"epochs": ep + 1, "val_auc": stop.best}
    for name, recs in evals.items():
        ids, p, y = predict(model, mk(recs, False), device)
        res[name] = dict(bag_ids=list(ids), y=[int(a) for a in y], p=[float(a) for a in p])
    del model, opt
    if device.type == "mps": torch.mps.empty_cache()
    return res


def main(folds, seed):
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config("configs/review1.yaml"); device = resolve_device("auto")
    bags = get_or_build_index(cfg)
    bs = get_or_make_split(bags, cfg.data.resolve_path("split_cache"), cfg.data.val_fraction, cfg.data.split_seed)
    cache = build_cache(bags, bs, Path("cache/final_cache.pkl"))
    allr = cache["train"] + cache["val"]; test = cache["test"]
    tv = sorted({r["bag_id"] for r in allr}); info = {b.bag_id: b for b in bags}
    y = np.array([info[b].label for b in tv]); g = np.array([info[b].patient_id for b in tv])
    sgkf = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(tv, y, g))
    print(f"{len(tv)} bags, {y.sum()} malignant, {len(set(g))} patients", flush=True)
    for k in folds:
        tr_i, te_i = sgkf[k]
        tr_b, ho_b = [tv[i] for i in tr_i], {tv[i] for i in te_i}
        # inner validation split (patient-grouped) for early stopping
        yi, gi = y[tr_i], g[tr_i]
        a, b_ = next(StratifiedGroupKFold(6, shuffle=True, random_state=k).split(tr_b, yi, gi))
        fit_b, val_b = {tr_b[i] for i in a}, {tr_b[i] for i in b_}
        sel = lambda S: [r for r in allr if r["bag_id"] in S]
        tr, va, ho = sel(fit_b), sel(val_b), sel(ho_b)
        tg, t0 = None, time.time()
        if any(VARIANTS[v].get("feat") for v in CV_VARIANTS):
            if not all((OUT / f"fold{k}_{v}.json").exists() for v in CV_VARIANTS if VARIANTS[v].get("feat")):
                teacher, tauc = train_teacher(tr, va, device)
                tg = make_targets(teacher, tr + va, device)
                print(f"[fold {k}] teacher val_auc={tauc:.3f} ({time.time()-t0:.0f}s)", flush=True)
                del teacher
        for v in CV_VARIANTS:
            f = OUT / f"fold{k}_{v}.json"
            if f.exists(): continue
            r = fit_student(v, seed, tr, va, {"heldout": ho, "test": test}, tg, device)
            r.update(variant=v, fold=k)
            json.dump(r, open(f, "w"))
            ho_m = classification_metrics(r["heldout"]["y"], r["heldout"]["p"])
            print(f"[fold {k}] {v:16s} heldout_auc={ho_m['auc']:.3f} epochs={r['epochs']} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4]); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(); main(a.folds, a.seed)
