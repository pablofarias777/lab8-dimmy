"""Pipeline base para treinamento com DPOTrainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer

REQUIRED_COLUMNS = ("prompt", "chosen", "rejected")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline base de DPO.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/hhh_preferences.jsonl"),
        help="Caminho do dataset no formato JSONL.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt2",
        help="Modelo base do ator.",
    )
    parser.add_argument(
        "--ref-model-name",
        type=str,
        default=None,
        help="Modelo de referencia congelado. Se omitido, usa o mesmo do ator.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/dpo-base",
        help="Diretorio de saida dos checkpoints.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Tamanho maximo de contexto usado no DPO.",
    )
    return parser.parse_args()


def load_preference_dataset(path: Path) -> Dataset:
    if not path.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {path}")

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            raw_line = line.strip()
            if not raw_line:
                continue
            item = json.loads(raw_line)
            if tuple(item.keys()) != REQUIRED_COLUMNS and set(item.keys()) != set(REQUIRED_COLUMNS):
                raise ValueError(
                    f"Linha {line_number} invalida. Esperado somente {REQUIRED_COLUMNS}, obtido {tuple(item.keys())}."
                )
            rows.append({key: item[key] for key in REQUIRED_COLUMNS})

    if not rows:
        raise ValueError("Dataset vazio. Adicione exemplos no arquivo JSONL.")

    return Dataset.from_list(rows)


def build_trainer(args: argparse.Namespace) -> DPOTrainer:
    dataset = load_preference_dataset(args.dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    actor_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    ref_model_name = args.ref_model_name or args.model_name
    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_name)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=20,
        learning_rate=5e-6,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=actor_model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        max_length=args.max_length,
    )
    return trainer


def main() -> None:
    args = parse_args()
    trainer = build_trainer(args)
    print("Pipeline DPO inicializado com sucesso.")
    print(f"Exemplos no dataset: {len(trainer.train_dataset)}")
    print("Pronto para executar trainer.train() no proximo passo.")


if __name__ == "__main__":
    main()
