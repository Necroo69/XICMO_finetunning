# Xicmo Fine-Tuning

Fine-tuning pipeline for the **Xicmo** family of specialized 8B agents.

A single base 8B model is adapted (LoRA/QLoRA) into a set of task-specialized
agents, each with its own dataset, training config, and evaluation harness.

## Stack

| Concern        | Tool                                   |
| -------------- | -------------------------------------- |
| Code & data    | GitHub                                 |
| Model weights  | Hugging Face Hub                       |
| Training       | RunPod (GPU)                           |

## Agents

Each agent lives under `agents/<name>/` and specializes the base model for one
channel or task:

- `seo` — search engine optimization content
- `geo` — generative engine optimization (LLM answer optimization)
- `instagram` — Instagram captions / posts
- `twitter` — Twitter/X posts and threads
- `reddit` — Reddit-style replies and posts
- `youtube` — YouTube titles, descriptions, scripts

> The roster is expected to grow to ~12 agents. To add one, copy an existing
> agent folder, rename it, and drop in its dataset + config.

## Layout

```
xicmo_finetuning/
├── agents/
│   └── <agent>/
│       ├── dataset/        # train/val/test data (git-ignored contents)
│       ├── config/         # <agent>.yaml training config
│       ├── train.py        # fine-tune the base model for this agent
│       └── evaluate.py     # evaluate a trained adapter/checkpoint
├── shared/
│   ├── data_utils.py       # dataset loading / formatting helpers
│   ├── eval_utils.py       # metrics + generation helpers
│   ├── upload_to_hf.py     # push adapters/weights to Hugging Face Hub
│   └── setup.py            # installable `xicmo_shared` package
├── experiments/
│   └── logs/               # run logs, metrics (git-ignored contents)
├── requirements.txt
├── .env.example
└── .gitignore
```

## Quick start

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e shared            # exposes `xicmo_shared` to every agent

# 2. Secrets
cp .env.example .env             # then fill in HF_TOKEN, WANDB_API_KEY, ...

# 3. Train an agent
python agents/seo/train.py --config agents/seo/config/seo.yaml

# 4. Evaluate
python agents/seo/evaluate.py --config agents/seo/config/seo.yaml

# 5. Publish weights
python shared/upload_to_hf.py --path experiments/seo/adapter --repo xicmo/seo-8b
```

## RunPod

1. Launch a GPU pod (e.g. A100 80GB) with a recent PyTorch/CUDA image.
2. Clone this repo, `pip install -r requirements.txt && pip install -e shared`.
3. Copy `.env` (or set env vars) with `HF_TOKEN` so datasets/weights can sync.
4. Run the agent's `train.py`; checkpoints land in `experiments/<agent>/`.

Model weights and checkpoints are **not** committed — they live on the
Hugging Face Hub. See `.gitignore`.
