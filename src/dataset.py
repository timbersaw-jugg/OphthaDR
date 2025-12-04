import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path

class FundusDataset(Dataset):
    """
    Dataset that reads labels.csv and images from image_dir.
    Expects labels.csv to contain either column 'image' or 'image_id' and a label column 'level' or 'label'.
    If 'split' exists in CSV it will filter rows by the split argument.

    Auto-creates stratified splits and persists to a new CSV if 'split' is missing.
    """
    COMMON_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

    def __init__(self, config, split='train', transform=None, image_extension='.jpg', return_meta=False):
        self.image_dir = config['data']['image_dir']
        self.label_csv = config['data']['label_csv']
        self.image_extension = image_extension
        self.transform = transform
        self.return_meta = return_meta
        self.config = config

        df = pd.read_csv(self.label_csv)
        # Remove unnamed pandas columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        # Determine image id column
        if 'image' in df.columns:
            df['image_id'] = df['image'].astype(str)
        elif 'image_id' in df.columns:
            df['image_id'] = df['image_id'].astype(str)
        else:
            raise ValueError("labels.csv must contain 'image' or 'image_id' column")

        # Keep raw image id (as in CSV) for reference; we'll attempt to resolve actual file on disk later
        df['image_id_raw'] = df['image_id'].astype(str)

        # Normalize label column names if necessary
        if 'level' not in df.columns and 'label' in df.columns:
            df = df.rename(columns={'label': 'level'})

        # If split column exists, normalize it and filter
        if 'split' in df.columns:
            # Normalize values to lowercase & strip whitespace
            df['split'] = df['split'].astype(str).str.strip().str.lower()
            if split is not None:
                df_filtered = df[df['split'] == str(split).lower()].reset_index(drop=True)
            else:
                df_filtered = df.reset_index(drop=True)
            self.data = df_filtered
            print(f"[dataset] Using existing 'split' column: selected split='{split}', rows={len(self.data)}")
        else:
            # Create stratified splits and persist the CSV for reproducibility
            try:
                from sklearn.model_selection import train_test_split
            except Exception as e:
                raise RuntimeError("scikit-learn required to create splits automatically. Install scikit-learn or provide a CSV with 'split' column.") from e

            if 'level' not in df.columns:
                raise ValueError("labels.csv must contain 'level' or 'label' column for stratified splitting")

            seed = int(self.config.get('seed', 42))
            # Fractions can be taken from config or default to 0.75/0.15/0.10
            train_frac = float(self.config.get('data', {}).get('split', {}).get('train', 0.75))
            val_frac = float(self.config.get('data', {}).get('split', {}).get('val', 0.15))
            test_frac = float(self.config.get('data', {}).get('split', {}).get('test', 0.10))
            if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
                # fallback to defaults
                train_frac, val_frac, test_frac = 0.75, 0.15, 0.10

            labels = df['level'].astype(int).values
            rem = 1.0 - train_frac
            # First split train vs remainder
            train_idx, rem_idx = train_test_split(df.index.values, stratify=labels, test_size=rem, random_state=seed)
            # Now split remainder into val and test
            if rem > 0:
                val_relative = val_frac / rem
                val_idx, test_idx = train_test_split(rem_idx, stratify=labels[rem_idx], test_size=1.0 - val_relative, random_state=seed)
            else:
                val_idx, test_idx = [], []

            df.loc[train_idx, 'split'] = 'train'
            df.loc[val_idx, 'split'] = 'val'
            df.loc[test_idx, 'split'] = 'test'

            # Persist so future runs use same splits
            orig = Path(self.label_csv)
            out_csv = orig.with_name(orig.stem + ".with_splits.csv")
            df.to_csv(out_csv, index=False)
            print(f"[dataset] No 'split' found. Created stratified splits (seed={seed}) and saved: {out_csv}")

            # Filter requested split
            if split is not None:
                df_filtered = df[df['split'] == str(split).lower()].reset_index(drop=True)
            else:
                df_filtered = df.reset_index(drop=True)
            self.data = df_filtered
            print(f"[dataset] Selected split='{split}', rows={len(self.data)} (train/val/test totals: "
                  f"{(df['split']=='train').sum()}/{(df['split']=='val').sum()}/{(df['split']=='test').sum()})")

        # final reset index
        self.data = self.data.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def _find_image_path(self, image_id_raw):
        """
        Given the raw image id from CSV, try:
          1) image_id_raw as-is (if it contains an extension)
          2) basename + each of COMMON_EXTS
        Returns the first existing full path, and the resolved filename (basename+ext).
        """
        # Try as given (maybe it already contains .jpg/.jpeg/.png or subfolders)
        candidate = os.path.join(self.image_dir, image_id_raw)
        if os.path.exists(candidate):
            return candidate, os.path.basename(candidate)

        # Strip any extension the CSV might have and try common ones
        base = os.path.splitext(image_id_raw)[0]
        tried = []
        for ext in self.COMMON_EXTS:
            p = os.path.join(self.image_dir, base + ext)
            tried.append(p)
            if os.path.exists(p):
                return p, os.path.basename(p)

        # As a last resort, try case-insensitive search for basename in image_dir
        # (useful when filesystem differs in case-sensitivity)
        try:
            files = os.listdir(self.image_dir)
            for fname in files:
                if os.path.splitext(fname)[0].lower() == base.lower():
                    p = os.path.join(self.image_dir, fname)
                    return p, fname
        except Exception:
            # If listing fails (permissions), ignore and continue to error
            pass

        return None, tried

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_id_raw = str(row['image_id_raw'])
        img_path, resolved = self._find_image_path(image_id_raw)

        if img_path is None:
            # resolved contains list of tried paths in that case
            tried = resolved if isinstance(resolved, list) else [os.path.join(self.image_dir, image_id_raw)]
            raise FileNotFoundError(
                f"Image not found for id '{image_id_raw}'. Tried paths:\n  " + "\n  ".join(tried)
            )

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Determine label
        if 'level' in row.index:
            label = int(row['level'])
        elif 'label' in row.index:
            label = int(row['label'])
        else:
            raise ValueError("labels.csv must contain 'level' or 'label' column for class label")

        if self.transform:
            image = self.transform(image)

        if self.return_meta:
            meta = {'image_id_raw': image_id_raw, 'image_id': resolved, 'path': img_path}
            return image, label, meta

        return image, label
