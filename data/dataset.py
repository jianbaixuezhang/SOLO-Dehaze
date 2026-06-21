from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

ImageTransform = Callable[[Image.Image], torch.Tensor]

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTENSIONS


def _list_image_files(directory: Union[str, Path]) -> List[Path]:
    d = Path(directory)
    if not d.is_dir():
        raise FileNotFoundError(f"Directory does not exist or is not a folder: {d.resolve()}")
    files = sorted(p for p in d.iterdir() if p.is_file() and _is_image_file(p))
    if not files:
        raise FileNotFoundError(f"No image files found in directory: {d.resolve()}")
    return files


def _load_rgb(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


class UnpairedTrainDataset(Dataset):

    def __init__(
        self,
        root: Union[str, Path],
        transform_hazy: Optional[ImageTransform] = None,
        transform_gt: Optional[ImageTransform] = None,
        length: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        root = Path(root)
        self.hazy_paths = _list_image_files(root / "train" / "hazy")
        self.gt_paths = _list_image_files(root / "train" / "gt")
        self.transform_hazy = transform_hazy
        self.transform_gt = transform_gt
        self._length = length if length is not None else max(len(self.hazy_paths), len(self.gt_paths))
        self._seed = seed

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if self._seed is not None:
            rng = random.Random((self._seed + index) & 0xFFFFFFFF)
            hi = rng.randrange(len(self.hazy_paths))
            gi = rng.randrange(len(self.gt_paths))
        else:
            hi = random.randrange(len(self.hazy_paths))
            gi = random.randrange(len(self.gt_paths))

        hazy = _load_rgb(self.hazy_paths[hi])
        clear = _load_rgb(self.gt_paths[gi])

        if self.transform_hazy is not None:
            hazy = self.transform_hazy(hazy)
        else:
            hazy = torch.from_numpy(np.asarray(hazy, dtype=np.uint8)).permute(2, 0, 1).float().div_(255.0)

        if self.transform_gt is not None:
            clear = self.transform_gt(clear)
        else:
            clear = torch.from_numpy(np.asarray(clear, dtype=np.uint8)).permute(2, 0, 1).float().div_(255.0)

        return {"hazy": hazy, "clear": clear}


class PairedValDataset(Dataset):

    def __init__(
        self,
        root: Union[str, Path],
        transform_hazy: Optional[ImageTransform] = None,
        transform_gt: Optional[ImageTransform] = None,
    ) -> None:
        root = Path(root)
        hazy_dir = root / "val" / "hazy"
        gt_dir = root / "val" / "gt"
        hazy_files = {p.name: p for p in hazy_dir.iterdir() if p.is_file() and _is_image_file(p)}
        gt_files = {p.name: p for p in gt_dir.iterdir() if p.is_file() and _is_image_file(p)}
        names = sorted(set(hazy_files.keys()) & set(gt_files.keys()))
        if not names:
            raise FileNotFoundError(
                f"No paired images with identical filenames found in val. Please check {hazy_dir} and {gt_dir}."
            )
        self.pairs: List[Tuple[Path, Path, str]] = [
            (hazy_files[n], gt_files[n], n) for n in names
        ]
        self.transform_hazy = transform_hazy
        self.transform_gt = transform_gt

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str]]:
        hazy_path, gt_path, name = self.pairs[index]
        hazy = _load_rgb(hazy_path)
        clear = _load_rgb(gt_path)

        if self.transform_hazy is not None:
            hazy = self.transform_hazy(hazy)
        else:
            hazy = torch.from_numpy(np.asarray(hazy, dtype=np.uint8)).permute(2, 0, 1).float().div_(255.0)

        if self.transform_gt is not None:
            clear = self.transform_gt(clear)
        else:
            clear = torch.from_numpy(np.asarray(clear, dtype=np.uint8)).permute(2, 0, 1).float().div_(255.0)

        return {"hazy": hazy, "clear": clear, "name": name}


class TestDataset(Dataset):

    def __init__(
        self,
        root: Union[str, Path],
        transform_hazy: Optional[ImageTransform] = None,
        transform_gt: Optional[ImageTransform] = None,
    ) -> None:
        root = Path(root)
        hazy_dir = root / "test" / "hazy"
        self.gt_dir = root / "test" / "gt"
        self.hazy_paths = _list_image_files(hazy_dir)
        self.gt_paths: Dict[str, Path] = {}
        if self.gt_dir.is_dir():
            for p in self.gt_dir.iterdir():
                if p.is_file() and _is_image_file(p):
                    self.gt_paths[p.name] = p
        self.transform_hazy = transform_hazy
        self.transform_gt = transform_gt

    def __len__(self) -> int:
        return len(self.hazy_paths)

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str, bool, None]]:
        hazy_path = self.hazy_paths[index]
        name = hazy_path.name
        hazy = _load_rgb(hazy_path)

        if self.transform_hazy is not None:
            hazy_t = self.transform_hazy(hazy)
        else:
            hazy_t = torch.from_numpy(np.asarray(hazy, dtype=np.uint8)).permute(2, 0, 1).float().div_(255.0)

        gt_path = self.gt_paths.get(name)
        clear_t: Optional[torch.Tensor] = None
        has_gt = False
        if gt_path is not None:
            has_gt = True
            clear = _load_rgb(gt_path)
            if self.transform_gt is not None:
                clear_t = self.transform_gt(clear)
            else:
                clear_t = torch.from_numpy(np.asarray(clear, dtype=np.uint8)).permute(2, 0, 1).float().div_(255.0)

        return {"hazy": hazy_t, "clear": clear_t, "name": name, "has_gt": has_gt}


def collate_test_batch(batch: List[Dict[str, Any]]) -> Dict[str, Union[torch.Tensor, List, List[Optional[torch.Tensor]]]]:
    hazies = torch.stack([b["hazy"] for b in batch], dim=0)
    names = [b["name"] for b in batch]
    clears: List[Optional[torch.Tensor]] = [b["clear"] for b in batch]
    has_gt = torch.tensor([bool(b["has_gt"]) for b in batch], dtype=torch.bool)
    return {"hazy": hazies, "clear": clears, "name": names, "has_gt": has_gt}
