"""
anime_tagger/face_cluster.py — Stage 3: face identity clustering.

Implements SCRFD face detection + ArcFace embedding directly via ONNX Runtime —
NO insightface Python package required (and therefore no C compilation / MSVC).

Architecture mirrors the existing WD Tagger pattern:
  download ONNX files → run onnxruntime → pure numpy pre/post-processing.

Model packs (downloaded once to ./models/buffalo_s/ or ./models/buffalo_l/):
  buffalo_s — det_500m.onnx  (SCRFD-500M)  + w600k_mbf.onnx (MobileFaceNet)
  buffalo_l — det_10g.onnx   (SCRFD-10G)   + w600k_r50.onnx  (ResNet-50)

Hard constraints:
  - Only moves files within output/ — never touches input folder
  - Never overwrites files — uses collision rename convention (stem_1.ext, …)
"""

from __future__ import annotations

import csv
import shutil
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from sklearn.cluster import DBSCAN

from anime_tagger.utils.image_utils import _SUPPORTED_SUFFIXES

# ── Buffalo model registry ────────────────────────────────────────────────────
_BUFFALO_CDN = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/{name}.zip"
)
_BUFFALO_MODELS: dict[str, dict[str, str]] = {
    "buffalo_s": {"det": "det_500m.onnx", "rec": "w600k_mbf.onnx"},
    "buffalo_l": {"det": "det_10g.onnx",  "rec": "w600k_r50.onnx"},
}

# ── Standard 5-point ArcFace alignment template at 112×112 ───────────────────
_ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# ── CSV column order ──────────────────────────────────────────────────────────
_FC_CSV_FIELDS = [
    "original_filename",
    "source_folder",
    "destination_path",
    "person_folder",
    "face_detected",
    "face_confidence",
    "faces_in_image",
    "embedding_model",
    "cluster_id",
    "cluster_size",
    "nearest_neighbour_dist",
    "intra_cluster_confidence",
]


# ── Model download ────────────────────────────────────────────────────────────

def _ensure_buffalo_models(model_name: str, models_dir: Path) -> tuple[Path, Path]:
    """
    Return (det_onnx_path, rec_onnx_path), downloading and extracting the zip
    from InsightFace CDN on first use.
    """
    pack = _BUFFALO_MODELS[model_name]
    pack_dir = models_dir / model_name
    det_path = pack_dir / pack["det"]
    rec_path = pack_dir / pack["rec"]

    if det_path.exists() and rec_path.exists():
        return det_path, rec_path

    pack_dir.mkdir(parents=True, exist_ok=True)
    url = _BUFFALO_CDN.format(name=model_name)
    zip_path = pack_dir / f"{model_name}.zip"

    print(f"  Downloading InsightFace {model_name} from CDN…")
    urllib.request.urlretrieve(url, zip_path)

    print(f"  Extracting {model_name}…")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Extract only the two ONNX files we need (zip may contain extras)
        for member in zf.namelist():
            fname = Path(member).name
            if fname in (pack["det"], pack["rec"]):
                zf.extract(member, pack_dir)
                # Flatten: move to pack_dir root if nested in a sub-folder
                extracted = pack_dir / member
                if extracted != pack_dir / fname:
                    (pack_dir / fname).parent.mkdir(parents=True, exist_ok=True)
                    extracted.rename(pack_dir / fname)

    zip_path.unlink(missing_ok=True)
    return det_path, rec_path


# ── ONNX provider selection (shared with wd_tagger.py pattern) ───────────────

def _get_providers() -> list[str]:
    available = ort.get_available_providers()
    if "DmlExecutionProvider" in available:
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


# ── NMS ───────────────────────────────────────────────────────────────────────

def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.4) -> list[int]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-6)
        order = rest[iou <= iou_threshold]
    return keep


# ── SCRFD face detector ───────────────────────────────────────────────────────

class _SCRFDDetector:
    """
    SCRFD face detector (ONNX).
    Inputs: BGR uint8 image (any size).
    Returns: list of (bbox [4], kps [5,2] | None, score float) sorted by score.
    """

    _STRIDES = [8, 16, 32]
    _NUM_ANCHORS = 2
    _INPUT_SIZE = (640, 640)

    def __init__(self, model_path: Path, providers: list[str]) -> None:
        self._session = ort.InferenceSession(str(model_path), providers=providers)
        self._center_cache: dict = {}

    def detect(
        self, img_bgr: np.ndarray, threshold: float = 0.5
    ) -> list[tuple[np.ndarray, np.ndarray | None, float]]:
        h, w = img_bgr.shape[:2]
        ih, iw = self._INPUT_SIZE
        scale = min(ih / h, iw / w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(img_bgr, (new_w, new_h))
        blob_img = np.zeros((ih, iw, 3), dtype=np.float32)
        blob_img[:new_h, :new_w] = resized
        blob_img = (blob_img - 127.5) / 128.0
        blob = blob_img.transpose(2, 0, 1)[np.newaxis]  # [1, 3, H, W]

        input_name = self._session.get_inputs()[0].name
        net_outs = self._session.run(None, {input_name: blob})

        # 9 outputs → scores, bbox, kps per stride; 6 outputs → scores, bbox only
        use_kps = len(net_outs) == 9
        fmc = 3

        scores_list: list[np.ndarray] = []
        bboxes_list: list[np.ndarray] = []
        kpss_list: list[np.ndarray] = []

        for idx, stride in enumerate(self._STRIDES):
            scores = net_outs[idx].reshape(-1)              # [N]
            bbox_preds = net_outs[idx + fmc].reshape(-1, 4) * stride  # [N, 4]

            feat_h, feat_w = ih // stride, iw // stride
            key = (feat_h, feat_w, stride)
            if key not in self._center_cache:
                gy, gx = np.mgrid[0:feat_h, 0:feat_w]
                centers = np.stack([gx, gy], axis=-1).astype(np.float32)
                centers = (centers * stride).reshape(-1, 2)
                centers = np.tile(
                    centers[:, np.newaxis, :], (1, self._NUM_ANCHORS, 1)
                ).reshape(-1, 2)
                self._center_cache[key] = centers
            anchors = self._center_cache[key]

            pos = scores >= threshold
            if not pos.any():
                continue

            pos_scores = scores[pos]
            pos_bboxes = self._dist2bbox(anchors[pos], bbox_preds[pos])
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            if use_kps:
                kps_preds = net_outs[idx + fmc * 2].reshape(-1, 10) * stride  # [N, 10]
                pos_kpss = self._dist2kps(anchors[pos], kps_preds[pos])
                kpss_list.append(pos_kpss.reshape(-1, 5, 2))

        if not scores_list:
            return []

        all_scores = np.concatenate(scores_list)
        all_bboxes = np.concatenate(bboxes_list) / scale
        all_kpss = np.concatenate(kpss_list) / scale if kpss_list else None

        keep = _nms(all_bboxes, all_scores)
        results = []
        for i in keep:
            kps = all_kpss[i] if all_kpss is not None else None
            results.append((all_bboxes[i], kps, float(all_scores[i])))
        return sorted(results, key=lambda x: -x[2])

    @staticmethod
    def _dist2bbox(centers: np.ndarray, dists: np.ndarray) -> np.ndarray:
        """SCRFD bounding box decoding (distance-to-bbox format)."""
        x1 = centers[:, 0] - dists[:, 0]
        y1 = centers[:, 1] - dists[:, 1]
        x2 = centers[:, 0] + dists[:, 2]
        y2 = centers[:, 1] + dists[:, 3]
        return np.stack([x1, y1, x2, y2], axis=1)

    @staticmethod
    def _dist2kps(centers: np.ndarray, kps_preds: np.ndarray) -> np.ndarray:
        """SCRFD keypoint decoding (offset from anchor center)."""
        out = kps_preds.copy()
        out[:, 0::2] += centers[:, 0:1]
        out[:, 1::2] += centers[:, 1:2]
        return out


# ── ArcFace recognizer ────────────────────────────────────────────────────────

class _ArcFaceRecognizer:
    """
    ArcFace recognizer (ONNX).
    Aligns a face to 112×112 via a 5-point similarity transform then returns
    a L2-normalised 512-dim embedding.
    """

    def __init__(self, model_path: Path, providers: list[str]) -> None:
        self._session = ort.InferenceSession(str(model_path), providers=providers)

    def embed(self, img_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
        """
        Align the face described by *kps* (5×2 landmark array) and return
        its L2-normalised 512-dim ArcFace embedding.
        """
        M, _ = cv2.estimateAffinePartial2D(
            kps.astype(np.float32), _ARCFACE_DST, method=cv2.LMEDS
        )
        if M is None:
            # Fall back to least-squares if LMEDS fails (e.g. only 1 face point)
            M, _ = cv2.estimateAffinePartial2D(
                kps.astype(np.float32), _ARCFACE_DST
            )
        if M is None:
            return np.zeros(512, dtype=np.float32)

        face = cv2.warpAffine(img_bgr, M, (112, 112), borderValue=0.0)

        blob = (face.astype(np.float32) - 127.5) / 128.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # [1, 3, 112, 112]

        input_name = self._session.get_inputs()[0].name
        emb: np.ndarray = self._session.run(None, {input_name: blob})[0][0]  # [512]

        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb


# ── Safe move (collision-safe, within output/ only) ───────────────────────────

def _safe_move(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        stem, suffix = src.stem, src.suffix
        counter = 1
        while dst.exists():
            dst = dst_dir / f"{stem}_{counter}{suffix}"
            counter += 1
    shutil.move(str(src), str(dst))
    return dst


# ── Public API ────────────────────────────────────────────────────────────────

class FaceClusterer:
    """
    SCRFD + ArcFace face identity clusterer using buffalo ONNX models directly.
    No insightface Python package — just onnxruntime, numpy, opencv.
    """

    def __init__(
        self,
        model_name: str = "buffalo_s",   # or "buffalo_l"
        eps: float = 0.50,
        min_samples: int = 1,
        det_size: tuple = (640, 640),    # passed to SCRFD input size
    ) -> None:
        if model_name not in _BUFFALO_MODELS:
            raise ValueError(f"Unknown model '{model_name}'. Choose buffalo_s or buffalo_l.")
        self.model_name = model_name
        self.eps = eps
        self.min_samples = min_samples
        self._detector: _SCRFDDetector | None = None
        self._recognizer: _ArcFaceRecognizer | None = None
        self._output_dir: Path | None = None

    def _load(self) -> None:
        if self._detector is not None:
            return
        models_dir = Path(__file__).resolve().parent.parent / "models"
        det_path, rec_path = _ensure_buffalo_models(self.model_name, models_dir)
        providers = _get_providers()
        self._detector = _SCRFDDetector(det_path, providers)
        self._recognizer = _ArcFaceRecognizer(rec_path, providers)

    def _get_face_data(
        self, img_bgr: np.ndarray
    ) -> tuple[np.ndarray | None, float, int]:
        """
        Detect faces in *img_bgr* and embed the highest-confidence one.
        Returns (embedding, det_score, num_faces) or (None, 0.0, 0).
        """
        assert self._detector is not None and self._recognizer is not None
        faces = self._detector.detect(img_bgr)
        if not faces:
            return None, 0.0, 0
        _, kps, score = faces[0]   # already sorted by score descending
        if kps is None:
            return None, score, len(faces)
        emb = self._recognizer.embed(img_bgr, kps)
        return emb, score, len(faces)

    def run_directory(
        self,
        target_dir: Path,
        progress_cb=None,
    ) -> dict:
        """
        Cluster faces in top-level image files of target_dir.
        No-face images → unknown_face/ inside target_dir.
        Clustered images → person_001/, person_002/, … sorted by cluster size.

        Returns {clustered, people, no_face, csv_rows}.
        """
        self._load()

        # source_folder label for CSV (e.g. "faces", "cosplay/haruhi_suzumiya")
        if self._output_dir is not None:
            try:
                source_folder = str(
                    target_dir.relative_to(self._output_dir)
                ).replace("\\", "/")
            except ValueError:
                source_folder = target_dir.name
        else:
            source_folder = target_dir.name

        # Top-level image files only (non-recursive)
        files = [
            p for p in sorted(target_dir.iterdir())
            if p.is_file() and p.suffix.lower() in _SUPPORTED_SUFFIXES
        ]

        if not files:
            return {"clustered": 0, "people": 0, "no_face": 0, "csv_rows": []}

        if progress_cb:
            progress_cb(f"Clustering faces in {source_folder} ({len(files)} images)…")

        # ── Extract embeddings ────────────────────────────────────────────────
        face_data: list[tuple[Path, np.ndarray, float, int]] = []
        no_face_entries: list[tuple[Path, float, int]] = []

        for img_path in files:
            img_arr = np.fromfile(str(img_path), dtype=np.uint8)
            img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                no_face_entries.append((img_path, 0.0, 0))
                continue
            try:
                emb, score, num_faces = self._get_face_data(img_bgr)
            except Exception:  # noqa: BLE001
                no_face_entries.append((img_path, 0.0, 0))
                continue
            if emb is None:
                no_face_entries.append((img_path, score, num_faces))
            else:
                face_data.append((img_path, emb, score, num_faces))

        # ── Move no-face images ───────────────────────────────────────────────
        csv_rows: list[dict] = []
        for img_path, _score, num_faces in no_face_entries:
            dst = _safe_move(img_path, target_dir / "unknown_face")
            csv_rows.append({
                "original_filename": img_path.name,
                "source_folder": source_folder,
                "destination_path": str(dst),
                "person_folder": "unknown_face",
                "face_detected": False,
                "face_confidence": "",
                "faces_in_image": num_faces,
                "embedding_model": self.model_name,
                "cluster_id": "",
                "cluster_size": "",
                "nearest_neighbour_dist": "",
                "intra_cluster_confidence": "",
            })

        if not face_data:
            return {
                "clustered": 0,
                "people": 0,
                "no_face": len(no_face_entries),
                "csv_rows": csv_rows,
            }

        # ── DBSCAN clustering ─────────────────────────────────────────────────
        # Embeddings are already L2-normalised from _ArcFaceRecognizer.embed()
        emb_matrix = np.array([e for _, e, _, _ in face_data], dtype=np.float32)

        db = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="cosine")
        labels = db.fit_predict(emb_matrix)

        # Sort clusters by size descending → person_001 is the largest cluster
        unique_labels = [lbl for lbl in set(labels) if lbl >= 0]
        cluster_sizes = {lbl: int(np.sum(labels == lbl)) for lbl in unique_labels}
        sorted_clusters = sorted(unique_labels, key=lambda lbl: -cluster_sizes[lbl])
        cluster_to_person = {
            lbl: f"person_{i + 1:03d}" for i, lbl in enumerate(sorted_clusters)
        }

        # ── Nearest-neighbour distances within each cluster ───────────────────
        nn_dists: dict[int, float] = {}
        for lbl in unique_labels:
            idxs = np.where(labels == lbl)[0]
            if len(idxs) == 1:
                nn_dists[int(idxs[0])] = 0.0
            else:
                sub = emb_matrix[idxs]
                sim = sub @ sub.T
                np.fill_diagonal(sim, -2.0)
                max_sim = sim.max(axis=1)
                for local_i, global_i in enumerate(idxs):
                    nn_dists[int(global_i)] = float(1.0 - max_sim[local_i])

        # ── Move clustered images ─────────────────────────────────────────────
        for i, (img_path, _emb, det_score, num_faces) in enumerate(face_data):
            cluster_id = int(labels[i])
            person_folder = cluster_to_person.get(cluster_id, "unknown_face")
            dst = _safe_move(img_path, target_dir / person_folder)

            nn_dist = nn_dists.get(i, 0.0)
            conf_label = "high" if nn_dist < 0.25 else "medium" if nn_dist < 0.35 else "low"

            csv_rows.append({
                "original_filename": img_path.name,
                "source_folder": source_folder,
                "destination_path": str(dst),
                "person_folder": person_folder,
                "face_detected": True,
                "face_confidence": f"{det_score:.4f}",
                "faces_in_image": num_faces,
                "embedding_model": self.model_name,
                "cluster_id": cluster_id,
                "cluster_size": cluster_sizes.get(cluster_id, 1),
                "nearest_neighbour_dist": f"{nn_dist:.4f}",
                "intra_cluster_confidence": conf_label,
            })

        return {
            "clustered": len(face_data),
            "people": len(sorted_clusters),
            "no_face": len(no_face_entries),
            "csv_rows": csv_rows,
        }

    def run_all(
        self,
        output_dir: Path,
        progress_cb=None,
    ) -> dict:
        """
        Run face clustering on:
          1. output/faces/  — top-level files
          2. output/cosplay/**  — top-level files per character sub-folder

        Writes face_cluster_log.csv to output_dir.
        Returns {clustered, people_faces, people_cosplay, no_face}.
        """
        self._output_dir = output_dir

        all_rows: list[dict] = []
        total_clustered = 0
        total_people_faces = 0
        total_people_cosplay = 0
        total_no_face = 0

        faces_dir = output_dir / "faces"
        if faces_dir.exists():
            r = self.run_directory(faces_dir, progress_cb)
            total_clustered += r["clustered"]
            total_people_faces += r["people"]
            total_no_face += r["no_face"]
            all_rows.extend(r.get("csv_rows", []))

        cosplay_dir = output_dir / "cosplay"
        if cosplay_dir.exists():
            for char_dir in sorted(cosplay_dir.iterdir()):
                if not char_dir.is_dir() or char_dir.name == "unknown_face":
                    continue
                r = self.run_directory(char_dir, progress_cb)
                total_clustered += r["clustered"]
                total_people_cosplay += r["people"]
                total_no_face += r["no_face"]
                all_rows.extend(r.get("csv_rows", []))

        if all_rows:
            csv_path = output_dir / "face_cluster_log.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=_FC_CSV_FIELDS)
                writer.writeheader()
                writer.writerows(all_rows)

        return {
            "clustered": total_clustered,
            "people_faces": total_people_faces,
            "people_cosplay": total_people_cosplay,
            "no_face": total_no_face,
        }
