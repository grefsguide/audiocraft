import argparse
import json
from pathlib import Path
from typing import Any

import comet_ml
from audiocraft import train


def flatten_dict(
    data: dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> dict[str, Any]:
    """
    Flatten a nested config dictionary into one-level key-value pairs.

    We use this so Hydra/Dora config values can be logged to Comet ML as
    standard experiment parameters with readable dotted keys.
    """
    items: dict[str, Any] = {}

    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, sep=sep))
        else:
            if isinstance(value, (str, int, float, bool)) or value is None:
                items[new_key] = value
            else:
                items[new_key] = str(value)

    return items


def main() -> None:
    """
    Upload one finished AudioCraft run to Comet ML.

    This script reads the saved experiment config, history, TensorBoard files,
    and key artifacts from one Dora run and sends them to Comet so we can
    attach a shareable training log link in the report.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--sig", type=str, required=True)
    parser.add_argument("--project_name", type=str, default="musicgen-finetune")
    parser.add_argument("--workspace", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    xp = train.main.get_xp_from_sig(args.sig)
    xp_dir = Path(xp.folder)
    history_path = xp_dir / "history.json"

    experiment = comet_ml.start(
        project_name=args.project_name,
        workspace=args.workspace,
    )

    experiment.set_name(args.run_name or args.sig)
    experiment.log_parameter("xp_sig", args.sig)
    experiment.log_parameter("xp_dir", str(xp_dir.resolve()))

    cfg = getattr(xp, "cfg", None)
    if cfg is not None:
        try:
            from omegaconf import OmegaConf

            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        except Exception:
            cfg_dict = None

        if isinstance(cfg_dict, dict):
            experiment.log_parameters(flatten_dict(cfg_dict))

    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))

        for epoch_idx, row in enumerate(history, start=1):
            if not isinstance(row, dict):
                continue

            for stage_name in ["train", "valid", "evaluate", "generate"]:
                stage_metrics = row.get(stage_name)
                if isinstance(stage_metrics, dict):
                    metric_payload = {}
                    for metric_name, metric_value in stage_metrics.items():
                        if isinstance(metric_value, (int, float)):
                            metric_payload[f"{stage_name}/{metric_name}"] = metric_value
                    if metric_payload:
                        experiment.log_metrics(metric_payload, step=epoch_idx)

            epoch_value = row.get("epoch")
            if isinstance(epoch_value, (int, float)):
                experiment.log_metric("epoch", epoch_value, step=epoch_idx)

    event_files = list(xp_dir.rglob("events.out.tfevents*"))
    if event_files:
        experiment.log_tensorboard_folder(str(xp_dir))

    checkpoint_path = xp_dir / "checkpoint.th"
    if checkpoint_path.exists():
        experiment.log_asset(str(checkpoint_path), file_name="checkpoint.th")
    if history_path.exists():
        experiment.log_asset(str(history_path), file_name="history.json")

    print("Comet URL:", experiment.url)
    experiment.end()


if __name__ == "__main__":
    main()