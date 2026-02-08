# fasttext-tokens
Extract tokens for a given label from a FastText model.

```
uv run fasttext_tokens.py --help
usage: fasttext_tokens.py [-h] [-n TOP_N] model label

extract most telling tokens for a FastText label

positional arguments:
  model              path to the fasttext model (.bin)
  label              the label to analyze (e.g., __label__1)

options:
  -h, --help         show this help message and exit
  -n, --top_n TOP_N  number of tokens to extract
```
