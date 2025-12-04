# # src/output_writer.py
# import json
# import os
# import csv
# from typing import List, Dict, Any
# import numpy as np

# def _to_native(obj):
#     """Convert numpy scalars/arrays recursively to native Python types suitable for json."""
#     if isinstance(obj, np.generic):
#         return obj.item()
#     if isinstance(obj, np.ndarray):
#         return obj.tolist()
#     if isinstance(obj, dict):
#         return {str(k): _to_native(v) for k, v in obj.items()}
#     if isinstance(obj, (list, tuple)):
#         return [_to_native(x) for x in obj]
#     return obj

# def write_outputs_json(output_dir: str, records: List[Dict[str, Any]],
#                        write_ndjson: bool = True, write_csv_summary: bool = True) -> Dict[str, str]:
#     """
#     Write outputs into:
#       - outputs.json  (pretty JSON)
#       - outputs.ndjson (one JSON per line)
#       - outputs_summary.csv (image_id, dr_stage, dr_stage_label, gradcam_heatmap_path)
#     Returns dict with paths.
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     out_json = os.path.join(output_dir, "outputs.json")
#     out_ndjson = os.path.join(output_dir, "outputs.ndjson")
#     out_csv = os.path.join(output_dir, "outputs_summary.csv")

#     if not isinstance(records, list):
#         raise ValueError("records must be a list of dicts")

#     safe_recs = [_to_native(r if isinstance(r, dict) else dict(r)) for r in records]

#     # pretty JSON
#     with open(out_json, "w", encoding="utf-8") as fh:
#         json.dump(safe_recs, fh, indent=2, ensure_ascii=False)
#     print(f"[output_writer] Saved: {out_json}")

#     if write_ndjson:
#         with open(out_ndjson, "w", encoding="utf-8") as fh:
#             for r in safe_recs:
#                 fh.write(json.dumps(r, ensure_ascii=False) + "\n")
#         print(f"[output_writer] Saved NDJSON: {out_ndjson}")

#     if write_csv_summary:
#         fieldnames = ["image_id", "dr_stage", "dr_stage_label", "gradcam_heatmap_path"]
#         # add any extra keys from first record
#         if len(safe_recs) > 0:
#             first_keys = [k for k in safe_recs[0].keys() if k not in fieldnames]
#             fieldnames.extend(first_keys)

#         with open(out_csv, "w", newline="", encoding="utf-8") as fh:
#             writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
#             writer.writeheader()
#             for r in safe_recs:
#                 row = {}
#                 for k in fieldnames:
#                     v = r.get(k, "")
#                     if isinstance(v, (list, dict)):
#                         row[k] = json.dumps(v, ensure_ascii=False)
#                     else:
#                         row[k] = v
#                 writer.writerow(row)
#         print(f"[output_writer] Saved CSV summary: {out_csv}")

#     return {"json": out_json, "ndjson": out_ndjson if write_ndjson else None, "csv": out_csv if write_csv_summary else None}


# def make_record(image_id, dr_stage, confidence, heatmap_path, roi_bbox=None, meta=None, label_map: Dict[int, str] = None) -> Dict[str, Any]:
#     """
#     Build a record for a single prediction. If label_map provided, also add 'dr_stage_label'.
#     """
#     dr_stage_int = int(dr_stage) if dr_stage is not None else None
#     if confidence is None:
#         conf_list = []
#     else:
#         if hasattr(confidence, "tolist"):
#             conf_list = [float(x) for x in confidence.tolist()]
#         else:
#             conf_list = [float(x) for x in list(confidence)]

#     rec = {
#         "image_id": str(image_id),
#         "dr_stage": dr_stage_int,
#         "stage_confidence": conf_list,
#         "gradcam_heatmap_path": str(heatmap_path) if heatmap_path is not None else "",
#     }

#     if label_map:
#         label = label_map.get(dr_stage_int)
#         if label is None:
#             label = label_map.get(str(dr_stage_int))
#         if label is not None:
#             rec["dr_stage_label"] = str(label)

#     if roi_bbox is not None:
#         rec["roi_bbox"] = [int(x) for x in roi_bbox]
#     if meta is not None:
#         rec["meta"] = _to_native(meta)

#     return rec
# src/output_writer.py
import json
import os
import csv
from typing import List, Dict, Any
import numpy as np

def _to_native(obj):
    """Convert numpy scalars/arrays recursively to native Python types suitable for json."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(x) for x in obj]
    return obj

def write_outputs_json(output_dir: str, records: List[Dict[str, Any]],
                       write_ndjson: bool = True, write_csv_summary: bool = True) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    out_json = os.path.join(output_dir, "outputs.json")
    out_ndjson = os.path.join(output_dir, "outputs.ndjson")
    out_csv = os.path.join(output_dir, "outputs_summary.csv")

    if not isinstance(records, list):
        raise ValueError("records must be a list of dicts")

    safe_recs = [_to_native(r if isinstance(r, dict) else dict(r)) for r in records]

    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(safe_recs, fh, indent=2, ensure_ascii=False)
    print(f"[output_writer] Saved: {out_json}")

    if write_ndjson:
        with open(out_ndjson, "w", encoding="utf-8") as fh:
            for r in safe_recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[output_writer] Saved NDJSON: {out_ndjson}")

    if write_csv_summary:
        fieldnames = ["image_id", "dr_stage", "dr_stage_label", "gradcam_heatmap_path"]
        if len(safe_recs) > 0:
            first_keys = [k for k in safe_recs[0].keys() if k not in fieldnames]
            fieldnames.extend(first_keys)

        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in safe_recs:
                row = {}
                for k in fieldnames:
                    v = r.get(k, "")
                    if isinstance(v, (list, dict)):
                        row[k] = json.dumps(v, ensure_ascii=False)
                    else:
                        row[k] = v
                writer.writerow(row)
        print(f"[output_writer] Saved CSV summary: {out_csv}")

    return {"json": out_json, "ndjson": out_ndjson if write_ndjson else None, "csv": out_csv if write_csv_summary else None}


def make_record(image_id: str,
                dr_stage: int,
                confidence: List[float],
                heatmap_path: str = None,
                roi_bbox: List[int] = None,
                meta: Dict[str, Any] = None,
                label_map: Dict[int, str] = None,
                retina_mask_path: str = None,
                mask_path: str = None,
                gradcam_stats: Dict[str, Any] = None,
                overlay_thumb_b64: str = None,
                qa_flags: List[str] = None) -> Dict[str, Any]:
    """
    Build a record for a single prediction. Returns a dict ready for JSON/NDJSON.
    """
    dr_stage_int = int(dr_stage) if dr_stage is not None else None
    if confidence is None:
        conf_list = []
    else:
        if hasattr(confidence, "tolist"):
            conf_list = [float(x) for x in confidence.tolist()]
        else:
            conf_list = [float(x) for x in list(confidence)]

    rec = {
        "image_id": str(image_id),
        "dr_stage": dr_stage_int,
        "stage_confidence": conf_list,
        "gradcam_heatmap_path": str(heatmap_path) if heatmap_path is not None else "",
    }

    if label_map:
        label = label_map.get(dr_stage_int)
        if label is None:
            label = label_map.get(str(dr_stage_int))
        if label is not None:
            rec["dr_stage_label"] = str(label)

    if roi_bbox is not None:
        rec["roi_bbox"] = [int(x) for x in roi_bbox]
    if meta is not None:
        rec["meta"] = _to_native(meta)
    if retina_mask_path is not None:
        rec["retina_mask_path"] = str(retina_mask_path)
    if mask_path is not None:
        rec["mask_path"] = str(mask_path)
    if gradcam_stats is not None:
        rec["gradcam_stats"] = _to_native(gradcam_stats)
    if overlay_thumb_b64 is not None:
        rec["overlay_thumb_b64"] = str(overlay_thumb_b64)
    if qa_flags is not None:
        rec["qa_flags"] = [str(x) for x in qa_flags]

    return rec
