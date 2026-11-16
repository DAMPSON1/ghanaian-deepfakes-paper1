#!/usr/bin/env python3
"""ResNet18 source-discrimination control: authentic Ghanaian vs authentic FF++ face crops.

Important: Both classes MUST be authentic (real) images. The test asks whether dataset
source can be predicted without any synthetic/manipulated images.

Expected structure:
  data/ghanaian_real/<original_video_id>/frame001.jpg
  data/ffpp_real/<original_video_id>/frame001.jpg

All frames of a source video stay together in one split. For related videos that share
an identity or FF++ source-video family, put them under a common group directory or
supply --manifest with a shared group_id.

Manifest option: CSV containing path, source, group_id (optional subject_id). Source is
'ghanaian' or 'ffpp'; path can be absolute or relative to --data-root. Using subject_id
as group_id (rather than video_id) is recommended for Ghanaian data and for FF++ when
reliable identity linkage is available.
"""
import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from PIL import Image
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 classification_report, confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score,
                                 roc_curve)
    from sklearn.model_selection import GroupShuffleSplit
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms
except ImportError as e:
    sys.exit(f'Missing dependency: {e}. Run: pip install -r requirements.txt')

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
SOURCES = {'ghanaian': 0, 'ffpp': 1}


def make_records(args):
    records = []
    if args.manifest:
        with open(args.manifest, newline='', encoding='utf-8-sig') as f:
            r = csv.DictReader(f)
            if not {'path', 'source', 'group_id'} <= set(r.fieldnames or []):
                raise ValueError('Manifest needs columns: path,source,group_id')
            for row in r:
                src = row['source'].strip().lower()
                if src not in SOURCES:
                    raise ValueError(f'Unknown source: {src}; allowed: ghanaian,ffpp')
                path = Path(row['path'])
                if not path.is_absolute():
                    path = args.data_root / path
                group = row['group_id'].strip()
                if not group:
                    raise ValueError(f'Empty group_id for {path}')
                records.append({'path': path.resolve(), 'source': src,
                                'group': f'{src}::{group}', 'group_id': group})
    else:
        for dirname, src in [('ghanaian_real', 'ghanaian'), ('ffpp_real', 'ffpp')]:
            root = args.data_root / dirname
            if not root.is_dir():
                raise ValueError(f'Missing directory: {root}')
            for path in sorted(root.rglob('*')):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                rel = path.relative_to(root)
                if len(rel.parts) < 2:
                    raise ValueError(
                        f'Flat image folder detected: {path}. Each source must have '
                        'subfolders named for original videos or identities; alternatively '
                        'provide a CSV via --manifest with verified group_id values.')
                group_id = rel.parts[0]
                records.append({'path': path.resolve(), 'source': src,
                                'group': f'{src}::{group_id}', 'group_id': group_id})
    if not records:
        raise ValueError('No images found')
    for record in records:
        if not record['path'].is_file():
            raise ValueError(f'Image not found: {record["path"]}')
    by_path = defaultdict(set)
    for r in records:
        by_path[str(r['path'])].add(r['source'])
    if any(len(v) > 1 for v in by_path.values()):
        raise ValueError('Same file was assigned to both sources')
    records = list({(str(r['path']), r['source']): r for r in records}.values())
    for src in SOURCES:
        groups = {r['group'] for r in records if r['source'] == src}
        if len(groups) < 5:
            raise ValueError(f'{src} has {len(groups)} independent groups; need >=5 for '
                             'train/validation/test group splitting.')
    return records


def split_records(records, seed):
    # Split independently within each source to ensure each partition has both classes.
    parts = {'train': [], 'val': [], 'test': []}
    for src in SOURCES:
        subset = [r for r in records if r['source'] == src]
        groups = np.array([r['group'] for r in subset])
        indices = np.arange(len(subset))
        s1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        trainval_idx, test_idx = next(s1.split(indices, groups=groups))
        s2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 17)
        train_sub, val_sub = next(s2.split(trainval_idx, groups=groups[trainval_idx]))
        for i in trainval_idx[train_sub]:
            parts['train'].append(subset[int(i)])
        for i in trainval_idx[val_sub]:
            parts['val'].append(subset[int(i)])
        for i in test_idx:
            parts['test'].append(subset[int(i)])
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        ga = {r['group'] for r in parts[a]}
        gb = {r['group'] for r in parts[b]}
        if ga & gb:
            raise RuntimeError(f'Group leakage between {a} and {b}')
    return parts


def choose_records(parts, max_images_per_group, seed):
    if max_images_per_group <= 0:
        return parts
    rng = random.Random(seed)
    capped = {}
    for split, recs in parts.items():
        groups = defaultdict(list)
        for r in recs:
            groups[r['group']].append(r)
        capped[split] = []
        for group, group_records in sorted(groups.items()):
            selected = rng.sample(group_records, min(len(group_records), max_images_per_group))
            capped[split].extend(selected)
    return capped


class FaceDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        with Image.open(r['path']) as im:
            rgb = im.convert('RGB')
        return self.transform(rgb), SOURCES[r['source']], i


def evaluate(model, loader, device):
    model.eval()
    y, probs, indices = [], [], []
    with torch.inference_mode():
        for batch, labels, ids in loader:
            logits = model(batch.to(device))
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
            y.extend(labels.numpy().tolist())
            indices.extend(ids.numpy().tolist())
    return np.array(y, dtype=int), np.array(probs), indices


def metrics_at(y, probs, threshold=0.5):
    pred = (probs >= threshold).astype(int)
    m = {'n_images': len(y), 'accuracy': float(accuracy_score(y, pred)),
         'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
         'precision_ffpp': float(precision_score(y, pred, zero_division=0)),
         'recall_ffpp': float(recall_score(y, pred, zero_division=0)),
         'f1_ffpp': float(f1_score(y, pred, zero_division=0)),
         'confusion_matrix_rows_actual_cols_pred_ghanaian_ffpp':
            confusion_matrix(y, pred, labels=[0, 1]).tolist(),
         'threshold': threshold}
    m['roc_auc_ffpp'] = float(roc_auc_score(y, probs)) if len(set(y)) == 2 else None
    return m


def write_csv(path, rows, columns):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, help='CSV with path,source,group_id (recommended)')
    ap.add_argument('--output-dir', type=Path, default=Path('source_discrimination_results'))
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-images-per-group', type=int, default=200,
                    help='Cap frames per group, independently per split. 0 = all frames.')
    ap.add_argument('--no-pretrained', action='store_true',
                    help='Train from random initialization (pretrained ImageNet default).')
    args = ap.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        ap.error('--epochs and --batch-size must be positive')
    args.data_root = args.data_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    records = make_records(args)
    parts = choose_records(split_records(records, args.seed), args.max_images_per_group, args.seed)
    counts = {}
    for split, recs in parts.items():
        counts[split] = {
            'n_images': len(recs),
            'n_groups': len({r['group'] for r in recs}),
            'by_source': dict(Counter(r['source'] for r in recs))}
        print(split, counts[split])
        if set(r['source'] for r in recs) != set(SOURCES):
            raise ValueError(f'Both sources must be present in {split}')
    write_csv(args.output_dir / 'split_manifest.csv', [
        {'split': split, 'path': str(r['path']), 'source': r['source'],
         'group_id': r['group_id']} for split, recs in parts.items() for r in recs
    ], ['split', 'path', 'source', 'group_id'])

    # Same geometry and normalization for BOTH classes, without augmentation
    # that would risk systematically altering manipulation/source artifacts.
    tx = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                  std=(0.229, 0.224, 0.225))])
    loaders = {split: DataLoader(FaceDataset(recs, tx), batch_size=args.batch_size,
                                  shuffle=(split == 'train'), num_workers=args.workers,
                                  pin_memory=torch.cuda.is_available())
               for split, recs in parts.items()}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    try:
        model = models.resnet18(weights=None if args.no_pretrained else models.ResNet18_Weights.DEFAULT)
    except Exception as e:
        raise RuntimeError('Unable to load pretrained weights. Check internet/cache, or '
                           'rerun with --no-pretrained for a random initialization.') from e
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)
    training_labels = [SOURCES[r['source']] for r in parts['train']]
    counts_train = np.bincount(training_labels, minlength=2)
    weights = torch.tensor(len(training_labels) / (2 * counts_train),
                           dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_samples = 0., 0
        for x, y, _ in loaders['train']:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(y)
            n_samples += len(y)
        val_y, val_p, _ = evaluate(model, loaders['val'], device)
        val_m = metrics_at(val_y, val_p)
        history.append({'epoch': epoch, 'train_loss': total_loss / n_samples,
                        'val_balanced_accuracy': val_m['balanced_accuracy'],
                        'val_auc': val_m['roc_auc_ffpp']})
        print(f'Epoch {epoch}/{args.epochs}: loss={total_loss/n_samples:.4f}, '
              f'val balanced accuracy={val_m["balanced_accuracy"]:.4f}, '
              f'val AUC={val_m["roc_auc_ffpp"]:.4f}')
        if val_m['balanced_accuracy'] > best_val:
            best_val = val_m['balanced_accuracy']
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch,
                        'seed': args.seed, 'class_mapping': SOURCES},
                       args.output_dir / 'best_model.pt')
    checkpoint = torch.load(args.output_dir / 'best_model.pt', map_location=device,
                            weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_y, test_p, idx = evaluate(model, loaders['test'], device)
    result = metrics_at(test_y, test_p)
    result['best_epoch'] = int(checkpoint['epoch'])
    result['class_mapping'] = SOURCES
    result['split_counts'] = counts
    result['seed'] = args.seed
    result['max_images_per_group'] = args.max_images_per_group
    result['pretrained'] = not args.no_pretrained
    result['note'] = ('Source discrimination among real images is not a direct measure '
                      'of which features the original deepfake classifier relied on.')
    (args.output_dir / 'test_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    rows = []
    for y, p, i in zip(test_y, test_p, idx):
        r = parts['test'][i]
        rows.append({'path': str(r['path']), 'source': r['source'],
                     'group_id': r['group_id'], 'true_label': int(y),
                     'probability_ffpp': f'{p:.10f}', 'predicted_label': int(p >= 0.5)})
    write_csv(args.output_dir / 'test_predictions.csv', rows,
              ['path', 'source', 'group_id', 'true_label', 'probability_ffpp', 'predicted_label'])
    write_csv(args.output_dir / 'training_history.csv', history,
              ['epoch', 'train_loss', 'val_balanced_accuracy', 'val_auc'])
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        cm = np.array(result['confusion_matrix_rows_actual_cols_pred_ghanaian_ffpp'])
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=['Ghanaian real', 'FF++ real']).plot(
            ax=ax, cmap='Blues', colorbar=False, values_format='d')
        fig.tight_layout()
        fig.savefig(args.output_dir / 'confusion_matrix.png', dpi=220)
        plt.close(fig)
        fpr, tpr, _ = roc_curve(test_y, test_p)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f'AUC = {result["roc_auc_ffpp"]:.4f}')
        ax.plot([0, 1], [0, 1], '--', label='Chance')
        ax.set(xlabel='False positive rate', ylabel='True positive rate', title='Source discrimination ROC')
        ax.legend(loc='lower right')
        fig.tight_layout()
        fig.savefig(args.output_dir / 'roc_curve.png', dpi=220)
        plt.close(fig)
    except ImportError:
        print('Matplotlib not available: skipped charts')
    print('\nFINAL HELD-OUT TEST METRICS')
    print(json.dumps(result, indent=2))
    print('Saved:', args.output_dir.resolve())


if __name__ == '__main__':
    main()