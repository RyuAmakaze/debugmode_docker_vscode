import types
import torch
import random
import math
from typing import Sequence, List
from torch.utils.data import Sampler
from tqdm import tqdm

try:
    from torchvision import datasets as tv_datasets, transforms as tv_transforms
except Exception:  # pragma: no cover - torchvision may not be installed
    class DummyCompose:
        def __init__(self, funcs):
            self.funcs = funcs
        def __call__(self, x):
            for f in self.funcs:
                x = f(x)
            return x

    class DummyLambda:
        def __init__(self, func):
            self.func = func
        def __call__(self, x):
            return self.func(x)

    class DummyToTensor:
        def __call__(self, x):
            return torch.tensor(x)

    tv_datasets = types.SimpleNamespace(MNIST=object, CIFAR10=object, CIFAR100=object)
    tv_transforms = types.SimpleNamespace(Compose=DummyCompose, Lambda=DummyLambda, ToTensor=DummyToTensor)

from config import (
    ENCODING_DIM,
    USE_DINO,
    DEVICE,
    PRELOAD_BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
)


def get_dataset_class(name: str):
    mapping = {
        "MNIST": tv_datasets.MNIST,
        "CIFAR10": tv_datasets.CIFAR10,
        "CIFAR100": tv_datasets.CIFAR100,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dataset: {name}")
    return mapping[name]


def _maybe_to_tensor(x):
    """Convert input to a tensor if it isn't already one."""
    if isinstance(x, torch.Tensor):
        return x
    return tv_transforms.ToTensor()(x)

from typing import Optional


class DinoEncoder:
    """Callable object that lazily loads a DINOv2 model for feature extraction."""

    def __init__(self):
        self.model = None
        self.preprocess = None
        self._fallback = False

    def _load(self):
        import os
        import torch.hub
        import fcntl
        import warnings
        from torchvision import transforms as _tt

        # Avoid race conditions when multiple workers load the model
        cache_dir = torch.hub.get_dir()
        os.makedirs(cache_dir, exist_ok=True)
        lock_path = os.path.join(cache_dir, "dinov2.lock")
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                self.model = torch.hub.load(
                    "facebookresearch/dinov2", "dinov2_vits14", pretrained=True
                )
            except Exception as e:  # pragma: no cover - network required
                warnings.warn(
                    f"Failed to load DINOv2 model ({e}); falling back to simple transform"
                )
                self._fallback = True
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

        if self._fallback:
            self.preprocess = _tt.Compose([
                _tt.Lambda(_maybe_to_tensor),
                _tt.Lambda(lambda x: x.view(-1)),
                _tt.Lambda(lambda x: x[:ENCODING_DIM]),
            ])
            return

        self.model.eval()
        self.model.to(DEVICE)

        self.preprocess = _tt.Compose(
            [
                _tt.Lambda(_maybe_to_tensor),
                _tt.Resize(224, antialias=True),
                _tt.CenterCrop(224),
                _tt.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __call__(self, img):
        if self._fallback:
            img = _maybe_to_tensor(img).view(-1)
            return img[:ENCODING_DIM]

        if self.model is None:
            self._load()

        if self._fallback:
            # _load() failed, use simple preprocessing
            img = _maybe_to_tensor(img).view(-1)
            return img[:ENCODING_DIM]

        img = self.preprocess(img)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        img = img.to(DEVICE)
        with torch.no_grad():
            feats = self.model(img)[0].cpu()
        return feats[:ENCODING_DIM]
def get_transform(use_dino: Optional[bool] = None):
    """Return a transform that converts images to feature vectors.

    Parameters
    ----------
    use_dino : bool, optional
        If ``True`` images are encoded using a pretrained DINOv2 model.
        This requires both ``torch`` and ``torchvision`` with the
        corresponding weights available.  When ``False`` the original
        behaviour of flattening the tensor and truncating to
        ``ENCODING_DIM`` is used.  When ``None`` (default) the value of
        ``config.USE_DINO`` is used.
    """

    if use_dino is None:
        use_dino = USE_DINO

    if not use_dino:
        return tv_transforms.Compose([
            tv_transforms.Lambda(_maybe_to_tensor),
            tv_transforms.Lambda(lambda x: x.view(-1)),
            tv_transforms.Lambda(lambda x: x[:ENCODING_DIM]),
        ])

    return DinoEncoder()


def filter_indices_by_class(dataset, num_classes):
    """Return indices of samples whose label is < num_classes."""
    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = dataset.labels
        targets = getattr(dataset, "labels", None)
    if targets is None and isinstance(dataset, torch.utils.data.TensorDataset):
        if len(dataset.tensors) < 2:
            raise ValueError(
                "TensorDataset must contain at least two tensors to provide labels"
            )
        targets = dataset.tensors[1]
    return [i for i, t in enumerate(targets) if int(t) < num_classes]


def compute_proportions(labels, num_classes):
    """Compute normalized label counts for a batch."""
    counts = torch.bincount(labels, minlength=num_classes).float()
    return counts / counts.sum()


class FixedBatchSampler(Sampler[List[int]]):
    """Yield predefined lists of indices as batches."""

    def __init__(self, batches: Sequence[Sequence[int]]):
        self.batches = [list(b) for b in batches]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def create_fixed_proportion_batches(dataset, teacher_probs_list, bag_size, num_classes):
    """Return a FixedBatchSampler where each batch matches the given proportions."""
    dataset_indices = list(range(len(dataset)))

    # Walk to the root dataset to access labels
    base_dataset = dataset
    while hasattr(base_dataset, "indices"):
        base_dataset = base_dataset.dataset

    targets = getattr(base_dataset, "targets", None)
    if targets is None:
        targets = getattr(base_dataset, "labels", None)
    if targets is None and isinstance(base_dataset, torch.utils.data.TensorDataset):
        if len(base_dataset.tensors) < 2:
            raise ValueError(
                "TensorDataset must contain at least two tensors to provide labels"
            )
        targets = base_dataset.tensors[1]
    if targets is None:
        raise ValueError(
            "Could not locate labels. Provide 'targets', 'labels', or use a TensorDataset with labels"
        )

    class_to_indices = {i: [] for i in range(num_classes)}
    for idx in dataset_indices:
        root_idx = idx
        ds = dataset
        # Resolve the index through potentially nested Subset objects
        while hasattr(ds, "indices"):
            root_idx = ds.indices[root_idx]
            ds = ds.dataset
        label = int(targets[root_idx])
        if label < num_classes:
            # store dataset-relative index
            class_to_indices[label].append(idx)

    for idx_list in class_to_indices.values():
        random.shuffle(idx_list)

    batches = []
    for probs in teacher_probs_list:
        raw = [p * bag_size for p in probs]
        counts = [math.floor(c) for c in raw]
        remaining = bag_size - sum(counts)
        fractions = [r - math.floor(r) for r in raw]
        for cls in sorted(range(num_classes), key=lambda i: fractions[i], reverse=True)[:remaining]:
            counts[cls] += 1

        batch = []
        for cls, count in enumerate(counts):
            batch.extend(class_to_indices[cls][:count])
            class_to_indices[cls] = class_to_indices[cls][count:]
        batches.append(batch)

    return FixedBatchSampler(batches)


def create_random_bags(dataset, bag_size, num_classes, shuffle=True):
    """Create random bags and return a sampler and teacher label proportions."""
    dataset_indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(dataset_indices)

    # Walk to the root dataset to access labels
    base_dataset = dataset
    while hasattr(base_dataset, "indices"):
        base_dataset = base_dataset.dataset

    targets = getattr(base_dataset, "targets", None)
    if targets is None:
        targets = getattr(base_dataset, "labels", None)
    if targets is None and isinstance(base_dataset, torch.utils.data.TensorDataset):
        if len(base_dataset.tensors) < 2:
            raise ValueError(
                "TensorDataset must contain at least two tensors to provide labels"
            )
        targets = base_dataset.tensors[1]
    if targets is None:
        raise ValueError(
            "Could not locate labels. Provide 'targets', 'labels', or use a TensorDataset with labels"
        )

    batches = []
    teacher_props = []
    # ignore last incomplete batch
    full_len = len(dataset_indices) - len(dataset_indices) % bag_size
    for start in range(0, full_len, bag_size):
        batch_indices = dataset_indices[start : start + bag_size]
        batches.append(batch_indices)

        labels = []
        for idx in batch_indices:
            root_idx = idx
            ds = dataset
            while hasattr(ds, "indices"):
                root_idx = ds.indices[root_idx]
                ds = ds.dataset
            label = int(targets[root_idx])
            if label < num_classes:
                labels.append(label)
        teacher_props.append(compute_proportions(torch.tensor(labels), num_classes))

    sampler = FixedBatchSampler(batches)
    teacher_tensor = torch.stack(teacher_props)
    return sampler, teacher_tensor


def preload_dataset(dataset, batch_size: int = PRELOAD_BATCH_SIZE, desc: str = "Preloading dataset"):
    """Load all features and labels into memory with progress output."""
    from torch.utils.data import DataLoader, TensorDataset

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=NUM_WORKERS > 0,
        multiprocessing_context="spawn",
    )

    all_x = []
    all_y = []
    total = len(loader)
    step = max(1, total // 10)

    print(desc)
    for i, (x, y) in tqdm(enumerate(loader, 1), desc="preload_dataset"):
        all_x.append(x.cpu())
        all_y.append(y.cpu())
        if i == 1 or i == total or i % step == 0:
            print(f"{desc} {i}/{total} batches")

    features = torch.cat(all_x)
    labels = torch.cat(all_y)
    return TensorDataset(features, labels)

def load_feature_dataset(path: str):
    """Load a ``TensorDataset`` saved by :func:`load_or_extract_features`."""
    from torch.utils.data import TensorDataset

    data = torch.load(path)
    if isinstance(data, dict):
        feats = data.get("features")
        labels = data.get("labels")
    else:
        feats, labels = data
    return TensorDataset(feats, labels)