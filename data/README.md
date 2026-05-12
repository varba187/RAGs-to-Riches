# Data

No manual download is required. All datasets are loaded automatically within the notebooks using the Hugging Face `datasets` library.

- **Natural Questions (NQ):** Loaded via `load_dataset("sentence-transformers/natural-questions")`. Used as the primary question answering dataset (10,000 training examples, 1,000 evaluation examples).
- **SQuAD:** Loaded via `load_dataset("squad", split="train[:10000]")`. Used as the passage retrieval corpus in place of the original Wikipedia/Wiki-DPR index, due to compute and storage constraints on Google Colab.

To reproduce, simply run the notebooks in order with an active internet connection and the required dependencies installed. See the main README for setup instructions.
