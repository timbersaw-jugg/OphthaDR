import yaml
from pathlib import Path
from copy import deepcopy

def load_config(base_path="configs/config.yaml", hpo_path="configs/config_hpo.yaml"):
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    if Path(hpo_path).exists():
        with open(hpo_path) as f:
            hpo = yaml.safe_load(f)
        def merge(a, b):
            for k, v in (b or {}).items():
                if k in a and isinstance(a[k], dict) and isinstance(v, dict):
                    merge(a[k], v)
                else:
                    a[k] = deepcopy(v)
        merge(cfg, hpo)
    cfg['data']['image_size'] = cfg.get('preprocessing', {}).get('resize', cfg['data'].get('image_size'))
    return cfg
