"""Pipeline de treinamento com DPOTrainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer

REQUIRED_COLUMNS = ("prompt", "chosen", "rejected")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline de DPO.")
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
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="Hiperparametro beta do DPO.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=1.0,
        help="Numero de epocas de treino.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
        help="Learning rate do otimizador.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Se informado, apenas inicializa o pipeline sem treinar.",
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
        gradient_accumulation_steps=4,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_steps=20,
        optim="paged_adamw_32bit",
        bf16=False,
        fp16=False,
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
        beta=args.beta,
    )
    return trainer


def main() -> None:
    args = parse_args()
    trainer = build_trainer(args)

    print("Pipeline DPO inicializado com sucesso.")
    print(f"Exemplos no dataset: {len(trainer.train_dataset)}")
    print(f"Beta configurado: {args.beta}")

    if args.skip_train:
        print("Treino nao executado (--skip-train).")
        return

    print("Iniciando treinamento DPO...")
    trainer.train()
    trainer.save_model(args.output_dir)
    trainer.processing_class.save_pretrained(args.output_dir)
    print(f"Treinamento finalizado. Modelo salvo em: {args.output_dir}")


if __name__ == "__main__":
    main()
