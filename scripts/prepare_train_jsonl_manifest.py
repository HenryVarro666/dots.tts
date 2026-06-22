#!/usr/bin/env python3

"""准备训练用 JSONL manifest 的脚本 / Build the training JSONL manifest.

本文件做什么 / What this file does:
    一键把公开数据集 **LJSpeech-1.1-48kHz**(LJSpeech 的 48 kHz 重制版)从
    Hugging Face Hub 拉下来、解包,再把它的 ``metadata.csv``(竖线 ``|`` 分隔)
    转成 dots.tts 训练数据流所需的 **JSONL manifest**:每行一个 JSON 对象,
    含 ``fid`` / ``audio`` / ``text`` 三个字段(``audio`` 为本地绝对路径)。
    最终切成 train / valid 两个 manifest 文件落盘。
    Downloads + extracts LJSpeech-48kHz, then emits train/valid JSONL manifests.

在训练数据流里的位置 / Where it sits in the pipeline:
    这是训练流水线的**最起点 / 离线预处理步骤**,只生成清单、不碰模型:
    它产出的 ``*.jsonl`` 正是 ``JsonlManifestSourceAdapter``
    (见 ``src/dots_tts/data/source_adapters/jsonl_manifest_adapter.py``)
    要读的输入 —— 那里按 ``fid`` / ``text`` / ``audio`` 这套 canonical 键名取值,
    再把 ``audio`` 路径喂给 BigVGAN AudioVAE 编成连续 latent、把 ``text`` 喂给
    Qwen2.5 backbone 的 tokenizer。所以本脚本写出的字段键名必须与 adapter 对齐。
    本脚本只依赖 ``huggingface_hub`` + 标准库,不需要 GPU。

关键函数 / Key functions:
    - ``parse_args`` —— 命令行参数(缓存/解包/输出目录、valid 集大小)。
    - ``main`` —— 下载→解包→读 csv→写两个 manifest 的全流程。

数据集约定 / Dataset assumptions:
    LJSpeech ``metadata.csv`` 每行格式为 ``fid|raw_text|normalized_text``;
    音频在 ``wavs/MossFormer2_SR_48K/<fid>.wav``(48 kHz 超分后的 wav)。
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download

# 仓库根目录:本文件位于 scripts/ 下,parents[1] 即上一级(repo 根)。
# Repo root = parent of the scripts/ directory holding this file.
REPO_ROOT = Path(__file__).resolve().parents[1]
# HF dataset 仓库 id 与压缩包文件名(48 kHz 重制版 LJSpeech)。
REPO_ID = "alibabasglab/LJSpeech-1.1-48kHz"
ARCHIVE_NAME = "LJSpeech-1.1-48kHz.tar.bz2"


def parse_args() -> argparse.Namespace:
    """解析命令行参数 / Parse CLI args.

    返回 (argparse.Namespace),含:
        --cache-dir   HF 下载缓存目录(压缩包落地处)。
        --extract-dir 解包目标目录(默认与 cache-dir 同处)。
        --output-dir  两个 manifest 的输出目录。
        --valid-size  取前 N 条作 validation 集,其余全部归 train(见 main)。
    默认目录都挂在 ``downloaded_data/`` 下,便于集中管理与 .gitignore。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "downloaded_data" / "hf_cache",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=REPO_ROOT / "downloaded_data" / "hf_cache",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "downloaded_data",
    )
    parser.add_argument("--valid-size", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    """下载→解包→读 metadata→写 train/valid manifest 的主流程 / Full pipeline.

    步骤 / Steps:
        1) 校验参数、把所有目录 resolve 成绝对路径并按需创建。
        2) 经 ``hf_hub_download`` 把数据集压缩包拉到 cache_dir(已缓存则跳过下载)。
        3) 若解包目录不存在则用系统 ``tar`` 解 bzip2(``-xjf``);存在则复用。
        4) 逐行读 ``metadata.csv``(竖线分隔),为每条记录拼出 ``fid/audio/text``
           并以 JSON 写出;前 ``valid_size`` 条进 valid manifest,其余进 train。
        5) 打印各路径与计数,便于核对。
    无返回值(产物是磁盘上的两个 .jsonl 文件 + stdout 报告)。
    """
    args = parse_args()
    if args.valid_size < 0:
        raise ValueError("valid_size must be >= 0")

    # 统一 resolve 成绝对路径:manifest 里写入的 audio 是绝对路径,训练侧
    # 无论从哪个 cwd 启动都能找到文件。
    # Resolve to absolute paths so the manifest's audio paths are cwd-independent.
    cache_dir = args.cache_dir.resolve()
    extract_dir = args.extract_dir.resolve()
    output_dir = args.output_dir.resolve()
    train_manifest_path = output_dir / "ljspeech_48khz_manifest_train.jsonl"
    valid_manifest_path = output_dir / "ljspeech_48khz_manifest_valid.jsonl"

    cache_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_path = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=ARCHIVE_NAME,
            local_dir=str(cache_dir),
        )
    )

    # 解包后数据集的根目录;存在即视为已解包,避免重复解压(幂等 / idempotent)。
    dataset_root = extract_dir / "LJSpeech-1.1-48kHz"
    if not dataset_root.exists():
        print("extracting archive...")
        # 用系统 tar 解 bzip2 压缩包:-x 解包 / -j bzip2 / -f 指定文件;
        # --checkpoint* 仅用于每 2000 条打印一次进度,check=True 在 tar
        # 非零退出时抛 CalledProcessError。
        subprocess.run(
            [
                "tar",
                "-xjf",
                str(archive_path),
                "-C",
                str(extract_dir),
                "--checkpoint=2000",
                "--checkpoint-action=echo=extracting...",
            ],
            check=True,
        )

    # metadata.csv:LJSpeech 转写清单;音频取 48 kHz 超分子目录(MossFormer2_SR_48K)。
    metadata_path = dataset_root / "metadata.csv"
    audio_dir = dataset_root / "wavs" / "MossFormer2_SR_48K"

    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_path}")
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"audio dir not found: {audio_dir}")

    train_count = 0
    valid_count = 0
    # 同时打开输入 csv 与两个输出 manifest;newline="" 是 csv 模块推荐写法,
    # 避免它和 universal-newline 处理冲突导致行解析异常。
    with (
        metadata_path.open("r", encoding="utf-8", newline="") as fin,
        train_manifest_path.open("w", encoding="utf-8") as train_fout,
        valid_manifest_path.open("w", encoding="utf-8") as valid_fout,
    ):
        reader = csv.reader(fin, delimiter="|")
        for row in reader:
            if not row:
                continue  # 跳过空行 / skip blank rows

            fid = row[0].strip()
            # LJSpeech 每行 = fid | raw_text | normalized_text。
            # 优先用第 3 列 normalized 文本(标点/数字已规范),缺失或为空时
            # 回退到第 2 列原始文本;normalized 通常更利于 TTS 训练。
            text = (
                row[2].strip() if len(row) >= 3 and row[2].strip() else row[1].strip()
            )
            # 音频写绝对路径(.resolve()),与上面把目录绝对化的目的一致。
            audio_path = (audio_dir / f"{fid}.wav").resolve()

            # 提前 fail-fast:清单里只允许出现确实存在的音频文件,
            # 避免训练时才在 dataloader 里炸。
            if not audio_path.is_file():
                raise FileNotFoundError(f"audio not found: {audio_path}")

            # ensure_ascii=False:非 ASCII 文本(如各语种字符)按原样写出而非 \uXXXX。
            record = json.dumps(
                {
                    "fid": fid,
                    "audio": str(audio_path),
                    "text": text,
                },
                ensure_ascii=False,
            )
            # 简单切分:遇到的前 valid_size 条进 valid,其余全进 train。
            # 因 metadata 行序固定,这是确定性的 split(非随机抽样)。
            if valid_count < args.valid_size:
                valid_fout.write(record)
                valid_fout.write("\n")  # JSONL:每条记录占一行
                valid_count += 1
            else:
                train_fout.write(record)
                train_fout.write("\n")
                train_count += 1

    print(f"archive: {archive_path}")
    print(f"dataset_root: {dataset_root}")
    print(f"train_manifest: {train_manifest_path}")
    print(f"valid_manifest: {valid_manifest_path}")
    print(f"train_records: {train_count}")
    print(f"valid_records: {valid_count}")


if __name__ == "__main__":
    main()
