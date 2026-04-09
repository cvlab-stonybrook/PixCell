"""Compare PixCell checkpoint output against Raj's validated reference images.

For each reference slide (A0RH, A0RT), generates:
1. PixCell probmap → thresholded tumor/TIL classification map
2. WSInfer classification map (ground truth)
3. Side-by-side comparison with Raj's reference image
4. Quantitative metrics: TIL FP rate, tumor accuracy, spatial correlation

Designed to run automatically after each eval checkpoint.

Usage:
  python virtual_staining/compare_with_reference.py \
    --checkpoint ~/checkpoints/til_probmap_lora_e2_s2750.pth \
    --eval_dir ~/data/eval_patches \
    --reference_dir ~/data/reference_images \
    --output_dir ~/outputs/reference_comparison
"""

import os
import csv
import argparse
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import json


def load_eval_results(eval_grid_path: str, manifest_path: str, n_patches: int = 17):
    """Extract per-patch results from an eval grid image."""
    grid = np.array(Image.open(eval_grid_path))
    h, w = grid.shape[:2]
    col_w = w // 3
    patch_h = 1024
    label_h = 30

    manifest = []
    with open(manifest_path) as f:
        manifest = list(csv.DictReader(f))

    results = []
    for i in range(min(n_patches, len(manifest), (h - label_h) // patch_h)):
        y = label_h + i * patch_h
        if y + patch_h > h:
            break
        results.append({
            'patch_id': manifest[i]['patch_id'],
            'slide': manifest[i].get('slide', ''),
            'he': grid[y:y+patch_h, 0:col_w],
            'gt': grid[y:y+patch_h, col_w:2*col_w],
            'gen': grid[y:y+patch_h, 2*col_w:3*col_w],
        })
    return results


def compute_metrics(gt: np.ndarray, gen: np.ndarray) -> dict:
    """Compute per-channel metrics between GT and generated probmaps."""
    gt_tumor = gt[:,:,0].astype(float)
    gt_til = gt[:,:,1].astype(float)
    gen_tumor = gen[:,:,0].astype(float)
    gen_til = gen[:,:,1].astype(float)

    # Tumor metrics
    tumor_mae = np.mean(np.abs(gen_tumor - gt_tumor))
    tumor_corr = np.corrcoef(gt_tumor.ravel(), gen_tumor.ravel())[0,1] if gt_tumor.std() > 0 else 0

    # TIL metrics
    til_mae = np.mean(np.abs(gen_til - gt_til))
    til_corr = np.corrcoef(gt_til.ravel(), gen_til.ravel())[0,1] if gt_til.std() > 0 else 0

    # TIL false positive rate: mean generated TIL where GT TIL is near zero
    pure_tumor_mask = (gt_tumor > 128) & (gt_til < 10)
    til_fp_rate = gen_til[pure_tumor_mask].mean() / 255.0 if pure_tumor_mask.sum() > 100 else -1
    til_fp_pct = (gen_til[pure_tumor_mask] > 20).mean() if pure_tumor_mask.sum() > 100 else -1

    # Thresholded accuracy
    gt_tumor_binary = gt_tumor > 128
    gen_tumor_binary = gen_tumor > 128
    tumor_accuracy = (gt_tumor_binary == gen_tumor_binary).mean()

    return {
        'tumor_mae': float(tumor_mae),
        'tumor_corr': float(tumor_corr),
        'til_mae': float(til_mae),
        'til_corr': float(til_corr),
        'til_fp_rate': float(til_fp_rate),
        'til_fp_pct': float(til_fp_pct),
        'tumor_accuracy': float(tumor_accuracy),
    }


def make_comparison_panel(
    he: np.ndarray, gt: np.ndarray, gen: np.ndarray,
    metrics: dict, label: str, size: int = 384,
) -> np.ndarray:
    """Create a comparison panel for one patch."""
    # Resize
    he_s = np.array(Image.fromarray(he).resize((size, size)))

    # GT and Gen tumor (red channel)
    gt_t = np.zeros((size, size, 3), dtype=np.uint8)
    gt_t[:,:,0] = np.array(Image.fromarray(gt[:,:,0]).resize((size, size), Image.NEAREST))
    gen_t = np.zeros((size, size, 3), dtype=np.uint8)
    gen_t[:,:,0] = np.array(Image.fromarray(gen[:,:,0]).resize((size, size), Image.NEAREST))

    # GT and Gen TIL (green channel)
    gt_l = np.zeros((size, size, 3), dtype=np.uint8)
    gt_l[:,:,1] = np.array(Image.fromarray(gt[:,:,1]).resize((size, size), Image.NEAREST))
    gen_l = np.zeros((size, size, 3), dtype=np.uint8)
    gen_l[:,:,1] = np.array(Image.fromarray(gen[:,:,1]).resize((size, size), Image.NEAREST))

    row1 = np.hstack([he_s, gt_t, gen_t])  # H&E, GT tumor, Gen tumor
    row2 = np.hstack([he_s, gt_l, gen_l])  # H&E, GT TIL, Gen TIL

    panel = np.vstack([row1, row2])

    # Add metrics text
    img = Image.fromarray(panel)
    d = ImageDraw.Draw(img)
    d.text((5, 5), label, fill=(255,255,255))
    d.text((size+5, 5), "GT Tumor", fill=(255,100,100))
    d.text((2*size+5, 5), f"Gen Tumor (corr={metrics['tumor_corr']:.2f})", fill=(255,100,100))
    d.text((5, size+5), f"TIL FP: {metrics['til_fp_pct']*100:.0f}%" if metrics['til_fp_pct'] >= 0 else "N/A", fill=(255,255,100))
    d.text((size+5, size+5), "GT TIL", fill=(100,255,100))
    d.text((2*size+5, size+5), f"Gen TIL (corr={metrics['til_corr']:.2f})", fill=(100,255,100))

    return np.array(img)


def main():
    parser = argparse.ArgumentParser(description="Compare checkpoint vs Raj's reference")
    parser.add_argument("--eval_grid", required=True, type=str,
                        help="Path to eval grid image from eval_til_lora.py")
    parser.add_argument("--manifest", required=True, type=str,
                        help="Path to eval patches manifest.csv")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--step_label", type=str, default="",
                        help="Step label for filenames (e.g., 'e2_s2750')")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load eval results
    results = load_eval_results(args.eval_grid, args.manifest)

    # Find reference slide patches
    ref_patches = [r for r in results if 'A0RH' in r['patch_id'] or 'A0RT' in r['patch_id']]

    print(f"Processing {len(ref_patches)} reference patches...")

    all_metrics = {}
    panels = []

    for r in ref_patches:
        metrics = compute_metrics(r['gt'], r['gen'])
        all_metrics[r['patch_id']] = metrics

        panel = make_comparison_panel(
            r['he'], r['gt'], r['gen'], metrics, r['patch_id'][:50]
        )
        panels.append(panel)

        print(f"  {r['patch_id'][:50]}:")
        print(f"    Tumor: MAE={metrics['tumor_mae']:.1f}, corr={metrics['tumor_corr']:.3f}, acc={metrics['tumor_accuracy']:.1%}")
        print(f"    TIL:   MAE={metrics['til_mae']:.1f}, corr={metrics['til_corr']:.3f}")
        if metrics['til_fp_pct'] >= 0:
            print(f"    TIL FP rate (in pure tumor): {metrics['til_fp_pct']:.1%}")

    # Aggregate metrics
    agg = {}
    for key in ['tumor_mae', 'tumor_corr', 'til_mae', 'til_corr', 'tumor_accuracy']:
        vals = [m[key] for m in all_metrics.values()]
        agg[key] = float(np.mean(vals))

    fp_vals = [m['til_fp_pct'] for m in all_metrics.values() if m['til_fp_pct'] >= 0]
    agg['til_fp_pct'] = float(np.mean(fp_vals)) if fp_vals else -1

    print(f"\n  AGGREGATE:")
    print(f"    Tumor corr: {agg['tumor_corr']:.3f}")
    print(f"    TIL corr: {agg['til_corr']:.3f}")
    print(f"    TIL FP rate: {agg['til_fp_pct']:.1%}" if agg['til_fp_pct'] >= 0 else "    TIL FP: N/A")

    # Save comparison panel
    if panels:
        combined = np.vstack(panels)
        out_path = args.output_dir / f"ref_comparison_{args.step_label}.jpg"
        Image.fromarray(combined).save(str(out_path), quality=95)
        print(f"\n  Saved: {out_path}")

    # Save metrics JSON (for tracking over time)
    metrics_path = args.output_dir / f"metrics_{args.step_label}.json"
    with open(metrics_path, 'w') as f:
        json.dump({'step': args.step_label, 'per_patch': all_metrics, 'aggregate': agg}, f, indent=2)

    # Append to running log
    log_path = args.output_dir / "metrics_log.csv"
    write_header = not log_path.exists()
    with open(log_path, 'a') as f:
        if write_header:
            f.write("step,tumor_corr,til_corr,til_fp_pct,tumor_mae,til_mae,tumor_accuracy\n")
        f.write(f"{args.step_label},{agg['tumor_corr']:.4f},{agg['til_corr']:.4f},"
                f"{agg['til_fp_pct']:.4f},{agg['tumor_mae']:.1f},{agg['til_mae']:.1f},"
                f"{agg['tumor_accuracy']:.4f}\n")
    print(f"  Metrics appended to: {log_path}")


if __name__ == "__main__":
    main()
