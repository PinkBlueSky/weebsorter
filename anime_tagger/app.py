"""
anime_tagger/app.py — Gradio UI and background classification worker.

ABSOLUTE SAFETY RULE: This module never deletes, moves, renames, or modifies
any file in the input folder or anywhere else on the filesystem.
Every file written goes to the designated output folder via safe_copy().
"""

from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import gradio as gr

from anime_tagger.classifiers.clip_classifier import CLIPClassifier
from anime_tagger.classifiers.wd_tagger import WDTagger
from anime_tagger.utils.file_ops import safe_copy, sanitize_folder_name
from anime_tagger.utils.image_utils import get_image_files, load_image_safe

try:
    from anime_tagger.face_cluster import FaceClusterer
    _FACE_CLUSTERING_AVAILABLE = True
except ImportError:
    _FACE_CLUSTERING_AVAILABLE = False

# ── Category → output subfolder name ─────────────────────────────────────────
_LABEL_DIR: dict[str, str] = {
    "real_face":    "faces",
    "landscape":    "landscapes",
    "architecture": "architecture",
    "food":         "food",
    "screenshot":   "screenshots",
    "abstract":     "abstract",
    "other":        "other",
    # anime_art is handled separately under anime/{series}/
}

# ── CSV column order ──────────────────────────────────────────────────────────
_CSV_FIELDS = [
    "original_filename",
    "original_path",
    "destination_path",
    "top_level_label",
    "clip_confidence",
    "detected_characters",
    "detected_series",
    "wd_rating",
    "wd_top10_tags",
]


@dataclass
class _LogEntry:
    original_filename: str
    original_path: str
    destination_path: str = ""
    top_level_label: str = ""
    clip_confidence: str = ""
    detected_characters: str = ""
    detected_series: str = ""
    wd_rating: str = ""
    wd_top10_tags: str = ""

    def as_dict(self) -> dict:
        return {
            "original_filename": self.original_filename,
            "original_path": self.original_path,
            "destination_path": self.destination_path,
            "top_level_label": self.top_level_label,
            "clip_confidence": self.clip_confidence,
            "detected_characters": self.detected_characters,
            "detected_series": self.detected_series,
            "wd_rating": self.wd_rating,
            "wd_top10_tags": self.wd_top10_tags,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_dur(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


# ── Shared processing state (written by worker thread, read by Gradio timer) ─
class _State:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reset_fields()

    def _reset_fields(self) -> None:
        self.running: bool = False
        self.stop_requested: bool = False
        self.phase: str = "idle"      # idle | loading | processing | clustering | done | stopped | error
        self.current: int = 0
        self.total: int = 0
        self.current_file: str = ""
        self.logs: list[_LogEntry] = []
        self.category_counts: dict[str, int] = {}
        self.skipped: list[tuple[str, str]] = []
        self.summary_md: str = ""
        self.error_msg: str = ""
        self.proc_start: float = 0.0
        self.img_times: list[float] = []
        self.face_stats: dict | None = None

    def reset(self) -> None:
        with self._lock:
            self._reset_fields()

    # -- convenience properties (lock-free, GIL provides enough safety for UI) --
    @property
    def progress_text(self) -> str:
        phase = self.phase
        if phase == "idle":
            return "Ready — click **▶ Start Processing** to begin."
        if phase == "loading":
            return f"⏳ Loading models… (`{self.current_file}`)"
        if phase == "processing":
            total = self.total or 1
            pct = int(100 * self.current / total)
            if self.img_times:
                avg = sum(self.img_times) / len(self.img_times)
                eta = _fmt_dur((self.total - self.current) * avg)
                timing = f" — {avg:.2f}s/img — ETA {eta}"
            else:
                timing = ""
            return (
                f"**Processing {self.current} / {self.total}** ({pct}%){timing}"
                f" — `{self.current_file}`"
            )
        if phase == "clustering":
            return "🔍 Running Stage 3 — face identity clustering…"
        if phase == "done":
            return f"✅ Done! Processed **{self.current}** image(s)."
        if phase == "stopped":
            return f"⏹ Stopped at **{self.current} / {self.total}**."
        if phase == "error":
            return f"❌ Error: {self.error_msg}"
        return ""


# ── CSV export ────────────────────────────────────────────────────────────────

def _write_csv(logs: list[_LogEntry], csv_path: Path) -> None:
    # SAFETY: copy only, never modify source — writes to output folder only
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for entry in logs:
            writer.writerow(entry.as_dict())


# ── Summary markdown ──────────────────────────────────────────────────────────

def _build_summary(state: _State, csv_path: Path) -> str:
    lines: list[str] = ["## Processing Summary\n"]
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    elapsed = time.monotonic() - state.proc_start if state.proc_start else 0.0
    avg_t = sum(state.img_times) / len(state.img_times) if state.img_times else 0.0
    lines.append(f"| Total processed | **{state.current}** |")
    lines.append(f"| Total skipped   | **{len(state.skipped)}** |")
    lines.append(f"| Total time      | **{_fmt_dur(elapsed)}** |")
    if avg_t:
        lines.append(f"| Avg per image   | **{avg_t:.2f}s** |")
        lines.append(f"| Fastest image   | **{min(state.img_times):.2f}s** |")
        lines.append(f"| Slowest image   | **{max(state.img_times):.2f}s** |")
    lines.append("")

    if state.category_counts:
        lines.append("### Per-category counts")
        lines.append("| Category | Count |")
        lines.append("|----------|-------|")
        for cat, cnt in sorted(state.category_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| `{cat}` | {cnt} |")
        lines.append("")

    if state.skipped:
        lines.append(f"### Skipped files ({len(state.skipped)} total)")
        for fname, reason in state.skipped[:30]:
            lines.append(f"- `{fname}` — {reason}")
        if len(state.skipped) > 30:
            lines.append(f"- *…and {len(state.skipped) - 30} more (see CSV log)*")
        lines.append("")

    lines.append(f"📄 **Classification log:** `{csv_path}`")

    if state.face_stats and state.face_stats.get("clustered", 0) > 0:
        fs = state.face_stats
        lines.append("")
        lines.append("### Stage 3 — Face Clustering")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Real people found    | **{fs.get('people_faces', 0)}** across `faces/` |")
        lines.append(f"| Cosplayers found     | **{fs.get('people_cosplay', 0)}** across `cosplay/` |")
        lines.append(f"| Images clustered     | **{fs.get('clustered', 0)}** |")
        lines.append(f"| No face detected     | **{fs.get('no_face', 0)}** → `unknown_face/` |")

    return "\n".join(lines)


# ── Background worker ─────────────────────────────────────────────────────────

def _run_worker(
    input_folder: Path,
    output_folder: Path,
    clip: CLIPClassifier,
    wd: WDTagger,
    state: _State,
    face_model: str = "buffalo_s",
) -> None:
    """
    Runs in a daemon thread.
    Classifies every supported image file found in input_folder (recursively),
    then copies the result to output_folder.

    ABSOLUTE SAFETY RULE: Only safe_copy() is used — the input folder and its
    contents are never modified, moved, renamed, or deleted.
    """
    try:
        # ── Load models ───────────────────────────────────────────────────────
        state.phase = "loading"
        state.current_file = "CLIP (openai/clip-vit-large-patch14)"
        clip.load()

        state.current_file = "WD Tagger (SmilingWolf/wd-vit-tagger-v3)"
        wd.load()

        # ── Discover images ───────────────────────────────────────────────────
        files = get_image_files(input_folder)
        state.total = len(files)
        state.phase = "processing"
        state.proc_start = time.monotonic()

        if not files:
            state.phase = "done"
            state.summary_md = "## No images found\nThe input folder contains no supported images."
            return

        # ── Classify + copy ───────────────────────────────────────────────────
        for i, img_path in enumerate(files):
            if state.stop_requested:
                state.phase = "stopped"
                break

            state.current = i + 1
            state.current_file = img_path.name
            img_t0 = time.monotonic()

            entry = _LogEntry(
                original_filename=img_path.name,
                original_path=str(img_path),
            )

            # Load image (handles GIF first-frame, corrupt files)
            img = load_image_safe(img_path)
            if img is None:
                reason = "Could not open or decode image (corrupted / unsupported)"
                state.skipped.append((img_path.name, reason))
                entry.top_level_label = "SKIPPED"
                entry.destination_path = f"[SKIPPED: {reason}]"
                state.logs.append(entry)
                continue

            try:
                # ── Stage 1: CLIP top-level classification ────────────────────
                label, conf = clip.classify(img)
                entry.top_level_label = label
                entry.clip_confidence = f"{conf:.4f}"

                # ── Stage 2: WD Tagger (anime art and cosplay) ───────────────
                if label in ("anime_art", "cosplay"):
                    wd_result = wd.tag(img)
                    entry.detected_characters = ";".join(wd_result["characters"])
                    entry.detected_series = ";".join(wd_result["series"])
                    entry.wd_rating = wd_result["rating"]
                    entry.wd_top10_tags = ";".join(
                        f"{t}:{s:.3f}" for t, s in wd_result["top10_tags"]
                    )

                    best = wd_result["best_series"]
                    if not best and wd_result["characters"]:
                        best = max(wd_result["characters"], key=wd_result["characters"].get)

                    if label == "anime_art":
                        series_dir = sanitize_folder_name(best) if best else "unknown"
                        dst_dir = output_folder / "anime" / series_dir
                        cat_key = f"anime/{series_dir}"
                    else:  # cosplay
                        series_dir = sanitize_folder_name(best) if best else "unknown_character"
                        dst_dir = output_folder / "cosplay" / series_dir
                        cat_key = f"cosplay/{series_dir}"
                else:
                    dst_dir = output_folder / _LABEL_DIR[label]
                    cat_key = label

                # ── Copy file to output ───────────────────────────────────────
                dst = safe_copy(img_path, dst_dir)  # SAFETY: copy only, never modify source
                entry.destination_path = str(dst)

                # Track per-category counts
                state.category_counts[cat_key] = (
                    state.category_counts.get(cat_key, 0) + 1
                )

            except Exception as exc:  # noqa: BLE001
                reason = str(exc)
                state.skipped.append((img_path.name, reason))
                entry.top_level_label = "ERROR"
                entry.destination_path = f"[ERROR: {reason}]"

            state.img_times.append(time.monotonic() - img_t0)
            state.logs.append(entry)

        # ── Export CSV ────────────────────────────────────────────────────────
        csv_path = output_folder / "classification_log.csv"
        _write_csv(state.logs, csv_path)  # SAFETY: copy only, never modify source

        # ── Stage 3: Face identity clustering ─────────────────────────────────
        if state.phase != "stopped" and _FACE_CLUSTERING_AVAILABLE:
            state.phase = "clustering"
            try:
                clusterer = FaceClusterer(model_name=face_model)
                state.face_stats = clusterer.run_all(output_folder)
            except Exception as exc:  # noqa: BLE001
                state.face_stats = {"error": str(exc)}

        if state.phase != "stopped":
            state.phase = "done"

        state.summary_md = _build_summary(state, csv_path)

    except Exception as exc:  # noqa: BLE001
        state.error_msg = str(exc)
        state.phase = "error"
    finally:
        state.running = False


# ── Gradio application ────────────────────────────────────────────────────────

def build_app(input_folder: Path, output_folder: Path) -> gr.Blocks:
    """Build and return the Gradio Blocks application."""

    clip = CLIPClassifier()
    wd = WDTagger()
    state = _State()

    # ── Event handlers ────────────────────────────────────────────────────────

    def _start(face_model: str) -> str:
        if state.running:
            return "⚠️ Already running — wait for completion or click Stop."
        state.reset()
        state.running = True
        t = threading.Thread(
            target=_run_worker,
            args=(input_folder, output_folder, clip, wd, state, face_model),
            daemon=True,
        )
        t.start()
        return "⏳ Starting…"

    def _stop(_: None = None) -> str:
        if not state.running:
            return state.progress_text
        state.stop_requested = True
        return "⏹ Stop requested — finishing current image…"

    def _poll() -> tuple[str, dict]:
        """Called every 500 ms by gr.Timer to update the UI."""
        text = state.progress_text
        finished = state.phase in ("done", "stopped", "error")
        summary_upd = (
            gr.update(value=state.summary_md, visible=True)
            if finished and state.summary_md
            else gr.update(visible=False)
        )
        return text, summary_upd

    # ── Layout ────────────────────────────────────────────────────────────────
    with gr.Blocks(title="Anime Tagger") as demo:
        gr.Markdown(
            "# 🎌 Anime Tagger\n"
            "Classifies images with CLIP then sub-tags anime art with WD-ViT-Tagger-v3."
        )

        with gr.Group():
            gr.Markdown(
                f"**Input folder** — `{input_folder}`  \n"
                f"**Output folder** — `{output_folder}`"
            )

        face_model_radio = gr.Radio(
            choices=[("Fast (buffalo_s, ~200MB)", "buffalo_s"), ("Accurate (buffalo_l, ~500MB)", "buffalo_l")],
            value="buffalo_s",
            label="Face model (used for Stage 3 face clustering)",
        )

        with gr.Row():
            start_btn = gr.Button("▶ Start Processing", variant="primary", scale=3)
            stop_btn  = gr.Button("⏹ Stop", variant="stop", scale=1)

        progress_md = gr.Markdown(
            "Ready — click **▶ Start Processing** to begin.",
            label="Status",
        )
        summary_md = gr.Markdown("", visible=False)

        # ── Timer: polls _poll() every 500 ms ─────────────────────────────────
        timer = gr.Timer(value=0.5)
        timer.tick(fn=_poll, outputs=[progress_md, summary_md])

        start_btn.click(fn=_start, inputs=[face_model_radio], outputs=[progress_md])
        stop_btn.click(fn=_stop,  inputs=[], outputs=[progress_md])

        gr.Markdown(
            "---\n"
            "**Safety**: all operations are copies — no source file is ever modified, "
            "moved, or deleted.\n"
            "Models are cached in `./models/` inside the project folder."
        )

    return demo
