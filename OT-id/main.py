#!/usr/bin/env python3
"""
OT-id — Operational Technology Identification
Penetration Testing Recon via Video Feed Analysis

Usage:
    # Analyze a TAK/RTSP drone feed:
    python main.py analyze --source rtsp --uri "rtsp://192.168.1.100:8554/live"

    # Analyze with CoT position from TAK server:
    python main.py analyze --source tak --uri "rtsp://192.168.1.100:8554/live" \
                           --tak-cot-host 0.0.0.0 --tak-cot-port 4242

    # Analyze an online stream (YouTube Live, NASA TV, etc.):
    python main.py analyze --source online --uri "https://www.youtube.com/watch?v=LIVE_ID"

    # Analyze a local video file:
    python main.py analyze --source file --uri "/path/to/recording.mp4"

    # Override CoT delivery target:
    python main.py analyze --source rtsp --uri "rtsp://..." \
                           --cot-host 192.168.1.50 --cot-port 6969
"""
import argparse
import os
import sys
import yaml
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # Overlay env vars
    cfg["shodan_api_key"] = os.environ.get("SHODAN_API_KEY", cfg.get("shodan_api_key", ""))
    cfg["nvd_api_key"] = os.environ.get("NVD_API_KEY", cfg.get("nvd_api_key", ""))
    cfg["anthropic_api_key"] = os.environ.get("ANTHROPIC_API_KEY", cfg.get("anthropic_api_key", ""))
    return cfg


def cmd_analyze(args):
    cfg = load_config(args.config)

    # CLI overrides
    if args.cot_host:
        cfg["cot_host"] = args.cot_host
    if args.cot_port:
        cfg["cot_port"] = args.cot_port

    from src.ingestion import get_source
    from src.detection.detector import DeviceDetector
    from src.detection.ocr import OCRReader
    from src.classification.classifier import DeviceClassifier
    from src.pipeline import OTidPipeline

    source_kwargs = {}
    if args.source in ("rtsp", "tak"):
        if args.tak_cot_host:
            source_kwargs["tak_cot_host"] = args.tak_cot_host
        if args.tak_cot_port:
            source_kwargs["tak_cot_port"] = args.tak_cot_port

    detector = DeviceDetector(
        model_path=cfg.get("yolo_model", "yolov8x.pt"),
        conf_threshold=cfg.get("detection_confidence", 0.35),
        device=cfg.get("inference_device", "cpu"),
    )
    ocr = OCRReader(gpu=cfg.get("ocr_gpu", False))
    classifier = DeviceClassifier(
        api_key=cfg.get("anthropic_api_key"),
        model=cfg.get("claude_model", "claude-haiku-4-5-20251001"),
    )

    pipeline = OTidPipeline(
        detector=detector,
        ocr=ocr,
        classifier=classifier,
        config=cfg,
        output_dir=os.path.join(BASE_DIR, "reports"),
        template_dir=os.path.join(BASE_DIR, "templates"),
        collect_training=args.collect_training,
    )

    frame_skip = cfg.get("frame_skip", 5)
    source = get_source(args.source, args.uri, frame_skip=frame_skip, **source_kwargs)

    print(f"\n[OT-id] Starting analysis")
    print(f"  Source type      : {args.source}")
    print(f"  URI              : {args.uri}")
    print(f"  CoT target       : {cfg.get('cot_host')}:{cfg.get('cot_port')}")
    print(f"  Model            : {cfg.get('yolo_model', 'yolov8x.pt')}")
    print(f"  Collect training : {args.collect_training}")
    print()

    max_frames = args.max_frames or cfg.get("max_frames", 0)
    frame_count = 0

    try:
        with source:
            for frame in source.frames():
                pipeline.process_frame(frame)
                frame_count += 1
                if max_frames and frame_count >= max_frames:
                    print(f"[OT-id] Reached max_frames={max_frames}, stopping.")
                    break
    except KeyboardInterrupt:
        print("\n[OT-id] Interrupted by user.")

    print(f"\n[OT-id] Processed {frame_count} frames.")
    result = pipeline.finalize(upload_to_roboflow=args.upload_roboflow)
    if result.get("device_count"):
        print(f"[OT-id] Identified {result['device_count']} unique devices.")
        print(f"  JSON report : {result['json']}")
        print(f"  HTML report : {result['html']}")
    if result.get("training"):
        t = result["training"]
        print(f"[OT-id] Training samples saved: {t['saved']}  →  {t['dataset_dir']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OT-id: Penetration Testing Recon via Video Feed Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default=os.path.join(BASE_DIR, "config.yaml"),
                        help="Path to config.yaml")

    sub = parser.add_subparsers(dest="command", required=True)

    # ── analyze ────────────────────────────────────────────────────────────────
    analyze = sub.add_parser("analyze", help="Analyze a video feed")
    analyze.add_argument("--source", required=True,
                         choices=["rtsp", "tak", "online", "file"],
                         help="Video source type")
    analyze.add_argument("--uri", required=True,
                         help="Stream URI or file path")
    analyze.add_argument("--tak-cot-host", default=None,
                         help="Bind address for CoT UDP listener (TAK position data)")
    analyze.add_argument("--tak-cot-port", type=int, default=4242,
                         help="UDP port for CoT telemetry (default 4242)")
    analyze.add_argument("--cot-host", default=None,
                         help="Override CoT delivery host")
    analyze.add_argument("--cot-port", type=int, default=None,
                         help="Override CoT delivery port")
    analyze.add_argument("--max-frames", type=int, default=0,
                         help="Stop after N frames (0 = unlimited)")
    analyze.add_argument("--collect-training", action="store_true",
                         help="Save classified detections as YOLO training data in data/training/")
    analyze.add_argument("--upload-roboflow", action="store_true",
                         help="Upload training data to Roboflow project after session (requires ROBOFLOW_API_KEY)")
    analyze.set_defaults(func=cmd_analyze)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
