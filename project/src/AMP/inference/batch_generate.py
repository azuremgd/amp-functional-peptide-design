# -*- coding: utf-8 -*-
import os, argparse, json, math, pickle, random
import pandas as pd
import torch
import numpy as np

from AMP.data_processing.data import AMPSeqDataset
from AMP.models.generator_model import AMPGenerator
from AMP.inference.generate import generate_with_constraints
from AMP.models.classifier import AMPActivity
from AMP.features.features import calculate_structural_features, FEATURE_KEYS
from AMP.data_processing.utils import max_identity


# --- Matplotlib 设置（无显示环境也能存图） ---
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

def make_plots(csv_path: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # (a) 活性分布
    if "activity_score" in df:
        plt.figure(figsize=(7,5))
        plt.hist(df["activity_score"], bins=30)
        plt.xlabel("Activity score"); plt.ylabel("Count")
        plt.title("Activity Score Distribution")
        plt.tight_layout(); plt.savefig(os.path.join(outdir, "activity_hist.png")); plt.close()

    # (a2) 广谱分布（可选）
    if "broad_score" in df and not df["broad_score"].isna().all():
        plt.figure(figsize=(7,5))
        plt.hist(df["broad_score"].dropna(), bins=30)
        plt.xlabel("Broadness score"); plt.ylabel("Count")
        plt.title("Broadness Score Distribution")
        plt.tight_layout(); plt.savefig(os.path.join(outdir, "broad_hist.png")); plt.close()

    # (b) 约束误差小提琴：actual - target
    errs, names = [], []
    for k in ["helix_propensity","disulfide_bonds","net_charge","hydrophobicity","length"]:
        ak, tk = f"actual_{k}", f"target_{k}"
        if ak in df and tk in df:
            errs.append((df[ak] - df[tk]).values)
            names.append(k)
    if errs:
        plt.figure(figsize=(8,5))
        plt.violinplot(errs, showextrema=True)
        plt.xticks(range(1, len(names)+1), names, rotation=20)
        plt.ylabel("actual - target")
        plt.title("Constraint Error (All Samples)")
        plt.tight_layout(); plt.savefig(os.path.join(outdir, "constraint_violin.png")); plt.close()

    # (c) 电荷 vs 疏水性，颜色=活性
    if {"actual_net_charge","actual_hydrophobicity","activity_score"} <= set(df.columns):
        plt.figure(figsize=(7,5))
        sc = plt.scatter(df["actual_net_charge"], df["actual_hydrophobicity"],
                         c=df["activity_score"], s=18)
        plt.xlabel("Net charge"); plt.ylabel("Hydrophobicity (GRAVY)")
        plt.title("Charge vs Hydrophobicity (color=activity)")
        cbar = plt.colorbar(sc); cbar.set_label("activity")
        plt.tight_layout(); plt.savefig(os.path.join(outdir, "charge_hydro_scatter.png")); plt.close()

    # (d) 长度：实际 vs 目标
    plt.figure(figsize=(7,5))
    if "length" in df:
        bins = np.arange(df["length"].min()-0.5, df["length"].max()+1.5)
        plt.hist(df["length"], bins=bins, alpha=0.7, label="actual")
    if "target_length" in df:
        bins_t = np.arange(df["target_length"].min()-0.5, df["target_length"].max()+1.5)
        plt.hist(df["target_length"], bins=bins_t, alpha=0.7, label="target")
    plt.legend(); plt.xlabel("Length"); plt.ylabel("Count")
    plt.title("Length Distribution")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "length_hist.png")); plt.close()

    # (e) 新颖性：与训练集的最大相似度
    if "max_identity" in df:
        plt.figure(figsize=(7,5))
        plt.hist(df["max_identity"], bins=30)
        plt.xlabel("Max identity vs training set"); plt.ylabel("Count")
        plt.title("Novelty (lower means newer)")
        plt.tight_layout(); plt.savefig(os.path.join(outdir, "identity_hist.png")); plt.close()

def sample_c_from_prior(prior, control=None, bias=None):
    # control: ["net_charge","hydrophobicity"]; bias 例: {"net_charge":"high","hydrophobicity":"low"}
    keys = prior["feature_keys"]
    mean, cov = prior["mean"], prior["cov"]
    p_lo, p_mid, p_hi = prior["p2_5"], prior["p50"], prior["p97_5"]

    def pick_dim(i, name):
        # 有偏置 -> 用分位区间抽样；否则高斯抽样后 clip
        if bias and name in bias:
            if bias[name] == "high":
                lo, hi = p_mid[i], p_hi[i]
            elif bias[name] == "low":
                lo, hi = p_lo[i], p_mid[i]
            else:
                lo, hi = p_lo[i], p_hi[i]
            return random.uniform(float(lo), float(hi))
        else:
            import numpy as np
            val = np.random.normal(mean[i], np.sqrt(cov[i,i]))
            return float(np.clip(val, p_lo[i], p_hi[i]))

    c = {}
    for i, name in enumerate(keys):
        c[name] = pick_dim(i, name)
    # 注：这里保留所有维度的采样值；如需“只控制部分维度”，可在这里决定是否覆盖/保留
    return c

def main(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv(args.data)
    ds = AMPSeqDataset(df, max_len=args.max_len)
    # ===== novelty reference cache（避免每条都处理 ds.seqs）=====
    refs = ds.seqs
    # 去重保持顺序（训练集可能有重复）
    seen = set()
    refs_uniq = []
    for s in refs:
        if s and s not in seen:
            seen.add(s)
            refs_uniq.append(s)

    # 固定只取前 N 条作为参考（与你的 limit 保持一致）
    novelty_refs = refs_uniq[:100]

    # 加载生成器
    # 加载生成器（重新训练后的 ckpt：包含 model + stats）
    if not os.path.isfile(args.generator):
        raise FileNotFoundError(f"generator ckpt not found: {args.generator}")
    if args.classifier and not os.path.isdir(args.classifier):
        print(f"[Warn] classifier dir not found: {args.classifier} (activity_score will be NA)")
        args.classifier = None
    ckpt = torch.load(args.generator, map_location=device)
    assert isinstance(ckpt, dict) and "model" in ckpt and "stats" in ckpt, \
        f"Bad generator checkpoint format: {type(ckpt)}, keys={list(ckpt.keys()) if isinstance(ckpt, dict) else None}"

    model = AMPGenerator().to(device)
    model.load_state_dict(ckpt["model"], strict=True)  # 重新训练后用 strict=True 最安全
    model.eval()
    stats = ckpt["stats"]


    # 加载二分类活性分类器（可选但建议）
    cls = None
    if args.classifier and os.path.isdir(args.classifier):
        cls = AMPActivity(device=device, model_dir=args.classifier)   # 关键：真正加载你训练好的 checkpoint_best
        cls.model.eval()
    else:
        print(f"[Warn] classifier dir not found: {args.classifier} (activity_score will be NA)")


    # 解析约束：优先用结构先验
    if args.use_prior:
        with open(args.use_prior, "rb") as f:
            prior = pickle.load(f)
        bias = dict(pair.split(":") for pair in args.bias.split(",")) if args.bias else None
        constraints_list = [ sample_c_from_prior(prior, control=None, bias=bias) for _ in range(args.groups) ]
    else:
        if args.constraints_file:
            with open(args.constraints_file, "r", encoding="utf-8") as f:
                constraints_list = json.load(f)
        else:
            constraints_list = json.loads(args.constraints)

    # 可选：提供 --total 时，覆盖 samples（向上取整均分到每组）
    if args.total is not None:
        per = int(math.ceil(args.total / max(1, len(constraints_list))))
        args.samples = per

    rows = []
    # ====== 采样 + “软引导 + 轻过滤”闭环 ======
    for gi, cons in enumerate(constraints_list):
        j = 0
        attempts = 0
        need = args.samples
        hard_len = int(cons['length'])
        while j < need and attempts < need * args.max_resample:
            attempts += 1
            seq = generate_with_constraints(
                model, stats, cons,
                max_length=hard_len,
                temperature=args.temperature, top_p=args.top_p, device=device
            )
            feats = calculate_structural_features(seq)

            # 轻量过滤（约束偏差过大再重采；阈值放宽一点）
            ok = True
            if abs(feats["length"] - cons["length"]) > 0.5: ok = False
            if feats["disulfide_bonds"] + 1e-6 < cons["disulfide_bonds"]: ok = False
            if abs(feats["net_charge"] - cons["net_charge"]) > args.charge_tol: ok = False
            if abs(feats["hydrophobicity"] - cons["hydrophobicity"]) > args.hydro_tol: ok = False
            if not ok:
                continue

            prob = cls.predict_prob(seq) if cls else None
            mi = max_identity(seq, novelty_refs, limit=len(novelty_refs))

            row = {
                'group': gi,
                'sample': j,
                'sequence': seq,
                'length': len(seq),
                'activity_score': prob,
                'max_identity': mi,
            }
            for k in FEATURE_KEYS:
                row[f'target_{k}'] = cons[k]
                row[f'actual_{k}'] = feats[k]

            # （占位）广谱分数：如果你已有多标签分类器，可在此接入并写入 row['broad_score']
            row['broad_score'] = None

            rows.append(row)
            act_str = f"{prob:.3f}" if prob is not None else "NA"
            print(f"[g{gi}#{j}] {seq} | act={act_str} | ident={mi:.3f}")
            j += 1
    # ====== 结束 ======

    # 保存 CSV
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    df_out = pd.DataFrame(rows)
    if args.min_broad_score is not None and "broad_score" in df_out.columns:
        df_out = df_out[df_out["broad_score"] >= args.min_broad_score]
    df_out.to_csv(args.out, index=False)
    print('Saved to', args.out)

    # 画图
    outdir = args.plots_outdir or os.path.splitext(args.out)[0] + "_plots"
    make_plots(args.out, outdir)
    print("Plots saved to", outdir)

if __name__ == '__main__':
    # 放在 argparse 之前
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../AMP
    def _p(*parts):  # project-root relative
        return os.path.join(PROJECT_ROOT, *parts)

    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default=_p('data', 'AMP-data.csv'),help="Reference dataset for novelty (max_identity). Default: data/AMP-data.csv")
    ap.add_argument('--generator', type=str, default=_p('checkpoints', 'generator_rl.pt'))
    ap.add_argument('--classifier', type=str, default=_p('checkpoints', 'cls', 'checkpoint_best'))


    # 手动 JSON 兼容
    ap.add_argument('--constraints', type=str, default='['
        '{"helix_propensity":1.2, "disulfide_bonds":2, "net_charge":2.0, "hydrophobicity":0.5, "length":20},'
        '{"helix_propensity":0.8, "disulfide_bonds":1, "net_charge":3.0, "hydrophobicity":0.3, "length":15}]')
    ap.add_argument('--constraints_file', type=str, default=None, help='JSON file with constraints')
    # 新增：结构先验
    ap.add_argument('--use_prior', type=str, default=None, help="path to prior.pkl (if set, ignore --constraints)")
    ap.add_argument('--bias', type=str, default=None, help="per-dim bias, e.g. 'net_charge:high,hydrophobicity:low'")
    ap.add_argument('--groups', type=int, default=10, help='number of groups (when using prior)')
    # 生成数量控制
    ap.add_argument('--samples', type=int, default=10)
    ap.add_argument('--total', type=int, default=None, help="可选，总生成条数；若给出，将均分到各约束组")
    # 采样策略
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--top_p', type=float, default=0.9)
    ap.add_argument('--max_len', type=int, default=64)
    # 过滤与输出
    ap.add_argument('--out', type=str, default='generated_amp_results.csv')
    ap.add_argument('--plots_outdir', type=str, default=None, help="图像输出目录（默认与CSV同名目录）")
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--charge_tol', type=float, default=0.7, help='允许的净电荷误差 |Δcharge|')
    ap.add_argument('--hydro_tol', type=float, default=0.4, help='允许的疏水性误差 |Δhydro|')
    ap.add_argument('--max_resample', type=int, default=20, help='单条最多重采次数')
    # 广谱（可选）
    ap.add_argument('--min_broad_score', type=float, default=None, help='filter by broadness score (0~1)')
    args = ap.parse_args()
    main(args)
