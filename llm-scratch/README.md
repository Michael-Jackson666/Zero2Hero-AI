# LLM Scratch Notes

This folder contains my working notes and Python experiments while studying large language models from first principles. The material closely follows the excellent open-source project ["LLMs-from-scratch" by Sebastian Raschka](https://github.com/rasbt/LLMs-from-scratch), with additional tweaks, comments, and experiments of my own.

## Repository Layout

- `chap2-work_with_text_data/`
	- Tokenization, byte pair encoding (BPE), and text preprocessing notebooks/scripts.
- `chap3-attention_mechanisms/`
	- Implementations of self-attention, masked attention, multi-head attention, and supporting utilities.
- `chap4-implement_gpt_model/`
	- Gradual build-up of a GPT-style transformer, including forward passes and generation routines.

Each chapter directory mirrors the structure of the upstream project while capturing my incremental implementations, test runs, and additional comments.

## How to Use

1. Create and activate a Python 3.10+ environment (conda or venv).
2. Install dependencies noted in the upstream guide, e.g. `pip install torch numpy matplotlib tqdm`.
3. Explore the chapter folders sequentially; scripts are organized so you can run them with `python <script>.py` and compare outputs against the original repository.

## Credits

Full credit for the learning path and reference implementation goes to Sebastian Raschka and contributors of the [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) project. This directory serves as my annotated fork for study purposes.

