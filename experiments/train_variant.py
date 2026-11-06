"""Train one (variant, seed) of the final ablation and dump bag-level predictions."""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from data.index import get_or_build_index
from data.splits import get_or_make_split
from data.transforms import IMAGENET_MEAN, IMAGENET_STD
from eval.metrics import aggregate_bag_predictions, classification_metrics
from experiments.cache import build_cache
from experiments.model import PlusNet
from models.teacher import TeacherNet
from training.engine import EarlyStopper
from training.losses import feature_distill_loss, kd_logit_loss
from utils.config import load_config
from utils.seed import resolve_device, set_seed

VARIANTS = {
    "baseline":        dict(),
    "kd":              dict(kd=1),
    "feat":            dict(feat=1),
    "kd_feat":         dict(kd=1, feat=1),                                   # Review-1 method
    "kd_feat_tcw":     dict(kd=1, feat=1, tcw=1),                            # + teacher-confidence weighting
    "clin":            dict(clin=1),                                         # descriptors only, no teacher
    "recon":           dict(recon=1),                                        # CEUS hallucination only
    "kd_feat_tcw_clin": dict(kd=1, feat=1, tcw=1, clin=1),
    "full":            dict(kd=1, feat=1, tcw=1, clin=1, recon=1),           # proposed B2CEUS-Net++
    # post-hoc (added after seeing the 9-variant test results): gentler combinations of the two best signals
    "feat_clin":       dict(feat=1, clin=1),
    "feat_clin_recon": dict(feat=1, clin=1, recon=1, w_clin=0.25, w_recon=0.1),
}
W = dict(kd=0.5, feat=1.0, clin=0.5, recon=1.0, T=4.0)
OUT = Path("results/final")


def tf(train: bool):
    ops = [A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0)]
    if train:
        ops += [A.HorizontalFlip(p=0.5),
                A.Affine(rotate=(-15, 15), scale=(0.9, 1.1), translate_percent=(0.0, 0.05), p=0.7),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.GaussNoise(std_range=(0.02, 0.08), p=0.2)]
    ops += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(ops, additional_targets={"ceus": "mask"})  # CEUS target gets the same geometry only


class CachedDS(Dataset):
    def __init__(self, recs, train):
        self.recs, self.t = recs, tf(train)

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        has = r["ceus"] is not None
        ceus = r["ceus"] if has else np.zeros_like(r["img"])
        o = self.t(image=np.stack([r["img"]] * 3, -1), ceus=ceus)
        c = F.adaptive_avg_pool2d(o["ceus"].float().div(255)[None, None], 56)[0]
        return dict(image=o["image"], ceus=c, has_ceus=torch.tensor(float(has)), label=torch.tensor(r["label"]),
                    desc=torch.from_numpy(r["desc"]), bag_id=r["bag_id"])


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ids, probs, ys = [], [], []
    for b in loader:
        p = F.softmax(model(b["image"].to(device))["logits"], 1)[:, 1].cpu().tolist()
        ids += b["bag_id"]; probs += p; ys += b["label"].tolist()
    return aggregate_bag_predictions(ids, probs, ys)


def teacher_targets(cfg, bags, bag_split, device, cache):
    path = Path("cache/teacher_targets.pt")
    if path.exists():
        return torch.load(path)
    from training.train_distill import build_teacher_targets
    from data.transforms import build_classification_transforms
    t = TeacherNet("resnet18", pretrained=False, fusion=cfg.distillation.teacher.fusion).to(device)
    t.load_state_dict(torch.load("checkpoints/distill/teacher_best.pt", map_location=device))
    tg = build_teacher_targets(t, bags, bag_split, build_classification_transforms(224, False), device)
    tg = {k: (v[0], v[1]) for k, v in tg.items()}
    torch.save(tg, path)
    return tg


def run(variant, seed, epochs, patience, save_ckpt):
    OUT.mkdir(parents=True, exist_ok=True)
    out_json = OUT / f"{variant}_s{seed}.json"
    if out_json.exists():
        print(f"skip {out_json.name}"); return
    cfg = load_config("configs/review1.yaml")
    v = VARIANTS[variant]
    set_seed(seed)
    device = resolve_device("auto")
    bags = get_or_build_index(cfg)
    bag_split = get_or_make_split(bags, cfg.data.resolve_path("split_cache"), cfg.data.val_fraction, cfg.data.split_seed)
    cache = build_cache(bags, bag_split, Path("cache/final_cache.pkl"))
    tg = teacher_targets(cfg, bags, bag_split, device, cache) if (v.get("kd") or v.get("feat")) else None

    mk = lambda s, tr: DataLoader(CachedDS(cache[s], tr), batch_size=16, shuffle=tr, num_workers=0)
    tr_l, va_l, te_l = mk("train", True), mk("val", False), mk("test", False)

    model = PlusNet(desc=bool(v.get("clin")), recon=bool(v.get("recon"))).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    stop = EarlyStopper(patience=patience, mode="max")
    best, hist, t0 = None, [], time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for b in tr_l:
            x, y = b["image"].to(device), b["label"].to(device)
            o = model(x)
            ce = F.cross_entropy(o["logits"], y, reduction="none")
            loss = ce.mean()
            if tg is not None:
                tl = torch.stack([tg[i][0] for i in b["bag_id"]]).to(device)
                tf_ = torch.stack([tg[i][1] for i in b["bag_id"]]).to(device)
                # teacher-confidence weight: probability the teacher gives the TRUE class
                w = F.softmax(tl, 1).gather(1, y[:, None])[:, 0] if v.get("tcw") else torch.ones_like(ce)
                if v.get("kd"):
                    kd = F.kl_div(F.log_softmax(o["logits"] / W["T"], 1), F.softmax(tl / W["T"], 1), reduction="none").sum(1) * W["T"] ** 2
                    loss = loss + W["kd"] * (w * kd).mean()
                if v.get("feat"):
                    fl = ((o["feats"] - tf_) ** 2).mean(1)
                    loss = loss + W["feat"] * (w * fl).mean()
            if v.get("clin"):
                loss = loss + v.get("w_clin", W["clin"]) * F.binary_cross_entropy_with_logits(o["desc"], b["desc"].to(device))
            if v.get("recon"):
                m = b["has_ceus"].to(device)
                if m.sum() > 0:
                    l1 = (o["ceus_hat"] - b["ceus"].to(device)).abs().mean((1, 2, 3))
                    loss = loss + v.get("w_recon", W["recon"]) * (l1 * m).sum() / m.sum()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * x.size(0); n += x.size(0)
        _, vp, vy = predict(model, va_l, device)
        vm = classification_metrics(vy, vp)
        improved = stop.step(vm["auc"])
        if improved:
            best = copy.deepcopy(model.state_dict())
        hist.append(dict(epoch=ep, loss=tot / n, val_auc=vm["auc"]))
        print(f"[{variant} s{seed}] ep{ep:02d} loss={tot/n:.3f} val_auc={vm['auc']:.3f}{' *' if improved else ''} ({time.time()-t0:.0f}s)", flush=True)
        if stop.should_stop:
            break
    model.load_state_dict(best)
    res = {}
    for name, ld in (("val", va_l), ("test", te_l)):
        ids, p, y = predict(model, ld, device)
        m = classification_metrics(y, p); m["bag_ids"] = ids
        res[name] = m
    res.update(variant=variant, seed=seed, history=hist, epochs_run=len(hist), seconds=time.time() - t0)
    json.dump(res, open(out_json, "w"))
    if save_ckpt:
        torch.save(best, OUT / f"{variant}_s{seed}.pt")
    print(f"[{variant} s{seed}] TEST auc={res['test']['auc']:.4f} acc={res['test']['accuracy']:.4f} f1={res['test']['f1']:.4f}", flush=True)
    del model, opt
    if device.type == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=5)
    a = ap.parse_args()
    for s in a.seeds:
        for vname in a.variants:
            run(vname, s, a.epochs, a.patience, save_ckpt=vname in ("baseline", "full", "feat_clin_recon"))
