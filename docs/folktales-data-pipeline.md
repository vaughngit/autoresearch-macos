# Folktales Data Pipeline

This branch is configured to train on the Hugging Face dataset:

```text
merve/folk-mythology-tales
```

The data-preparation code lives in `prepare.py`.

## What `prepare.py` Does

`prepare.py` prepares the dataset so `train.py` can consume it. It:

1. Downloads the Hugging Face Parquet file.
2. Keeps clean rows from the `text` column.
3. Creates local train and validation files.
4. Trains an 8,192-token BPE tokenizer.
5. Saves tokenizer support files for training and evaluation.
6. Exposes the dataloader and validation metric used by `train.py`.

During research experiments, treat `prepare.py` as fixed. The experiment surface
is `train.py`.

## Cache Location

Prepared data is stored under:

```text
~/.cache/autoresearch/folk-mythology-tales/
```

The prepared files are:

```text
data/raw_train.parquet   original downloaded Hugging Face parquet
data/train.parquet       local training split
data/val.parquet         local validation split
tokenizer/tokenizer.pkl  trained tokenizer
tokenizer/token_bytes.pt token byte-length lookup for BPB evaluation
```

## Train and Validation Split

The original dataset has a `train` split. This branch creates a local validation
split from that data.

The current split is:

```text
190,792 training rows
10,041 validation rows
```

Training rows teach the model. Validation rows are held back so the final score
can measure how well the model predicts text it did not directly train on.

## Key Terms

- **Dataset**: A collection of examples. Here, each example is folklore or
  mythology text.
- **Parquet**: A compact table file format, like a program-friendly spreadsheet.
- **Tokenizer**: The tool that turns text into numbers.
- **Token**: A chunk of text, such as a word, part of a word, punctuation, or
  whitespace.
- **Dataloader**: Code that feeds prepared batches into the model.
- **Validation**: Measuring the model on held-back data.
- **BPB**: Bits per byte. The validation score used by this project. Lower is
  better.
