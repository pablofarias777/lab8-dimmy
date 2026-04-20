# Laboratório 08 — Alinhamento Humano com DPO

> **Disciplina:** Inteligência Artificial Aplicada  
> **Instituição:** Instituto iCEV  
> **Aluno:** Pablo Ferreira de Andrade Farias  
> **Orientador:** Prof. Dimmy  
> **Entrega:** versão `v1.0`

---

> **Nota de Integridade Acadêmica:**  
> *"Partes geradas/complementadas com IA, revisadas por Pablo Ferreira de Andrade Farias"*

> **Uso de IA:**  
> Ferramentas de IA generativa foram usadas como apoio na estruturação e documentação do projeto. Todo o conteúdo foi revisado criticamente e validado pelo aluno antes da submissão.

---

## Objetivo

Este laboratório implementa um **pipeline de alinhamento de LLM com DPO (Direct Preference Optimization)** para reforçar comportamento **HHH**:

- **Helpful** (útil)
- **Honest** (honesto)
- **Harmless** (inofensivo)

Em vez de RLHF completo, foi usado DPO com pares de preferência no formato:

- `prompt`
- `chosen` (resposta segura/alinhada)
- `rejected` (resposta inadequada/prejudicial)

---

## Estrutura do Projeto

```text
lab8-dimmy/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── hhh_preferences.jsonl
├── src/
│   ├── train_dpo.py
│   └── infer.py
└── scripts/
    ├── run_train.sh
    └── run_infer.sh
```

---

## Como Executar

### 1. Criar ambiente virtual e instalar dependências

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Testar pipeline sem treinar

```bash
python3 src/train_dpo.py --skip-train
```

Isso valida carregamento de:
- dataset de preferências;
- tokenizer/modelo ator;
- modelo de referência;
- configuração do `DPOTrainer`.

### 3. Executar treino DPO

```bash
bash scripts/run_train.sh
```

Comando equivalente:

```bash
python3 src/train_dpo.py \
  --dataset-path data/hhh_preferences.jsonl \
  --model-name gpt2 \
  --output-dir outputs/dpo-lab8 \
  --beta 0.1
```

### 4. Executar validação de inferência

```bash
bash scripts/run_infer.sh
```

A validação compara a pontuação condicional de:
- resposta `chosen` (segura)
- resposta `rejected` (inadequada)

Resultado esperado no console: `score_chosen > score_rejected`.

---

## Explicação Técnica

### Passo 1 — Dataset de Preferências (HHH)

O arquivo `data/hhh_preferences.jsonl` foi construído no formato JSONL com as colunas obrigatórias:

```json
{"prompt":"...","chosen":"...","rejected":"..."}
```

O dataset contém exemplos de:
- solicitações maliciosas (fraude, invasão, abuso);
- adequação de tom corporativo (respostas profissionais e respeitosas).

### Passo 2 — Pipeline DPO

No `src/train_dpo.py`:

- `AutoModelForCausalLM` para modelo ator;
- segundo modelo como referência (congelado por padrão no fluxo do DPO);
- `DPOTrainer` da biblioteca `trl`;
- `DPOConfig` para hiperparâmetros de treino.

### Passo 3 — Papel Matemático do \(\beta\)

No DPO, o termo de preferência é regularizado por \(\beta\), que controla a intensidade da atualização em relação ao modelo de referência. Intuitivamente, \(\beta\) funciona como um **imposto de desvio**: quanto maior o incentivo para satisfazer preferências, mais o modelo tende a se afastar do comportamento linguístico original; o termo com \(\beta\) limita esse afastamento e evita perda de fluência/coerência. Neste laboratório foi usado `beta=0.1`, equilibrando alinhamento de segurança e preservação da qualidade textual.

### Passo 4 — Otimização e Execução

Configurações principais:

- `beta=0.1`
- `gradient_accumulation_steps=4`
- `learning_rate=5e-6`
- `num_train_epochs=1`
- otimizador automático por ambiente:
  - `paged_adamw_32bit` quando há CUDA
  - `adamw_torch` em CPU/Mac sem CUDA

---

## Dependências Principais

| Biblioteca | Uso no projeto |
|------------|----------------|
| `torch` | Base de treino/inferência |
| `transformers` | Modelos/tokenizer |
| `datasets` | Dataset em formato HF |
| `trl` | `DPOTrainer` e `DPOConfig` |
| `accelerate` | Suporte de execução |
| `bitsandbytes` | Otimizadores/eficiência quando suportado |
| `peft` | Extensões de fine-tuning (apoio) |

---

## Checklist de Entrega

- [x] Dataset de preferências com colunas `prompt`, `chosen`, `rejected`.
- [x] Pipeline DPO implementado com `DPOTrainer`.
- [x] Hiperparâmetro `beta=0.1` configurado.
- [x] Validação de inferência com comparação `chosen` vs `rejected`.
- [x] Nota de integridade acadêmica adicionada no README.
- [ ] Tag/release final publicada como `v1.0`.

Comandos de fechamento:

```bash
git add .
git commit -m "docs: finaliza readme do lab 08"
git push origin main
git tag -a v1.0 -m "Entrega final Lab 08"
git push origin v1.0
```

---

## Referências

- [Direct Preference Optimization (DPO) — Paper](https://arxiv.org/abs/2305.18290)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets)
