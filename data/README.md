# RAGs to Riches: A Reimplementation of RAG for Knowledge-Intensive NLP Tasks

This repository contains Cornell CS 4782 final group project, which is a reimplementation of **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** by Lewis et al. In this project, we reproduce a simplified version of the Natural Questions result from the original paper, focusing on whether retrieval-augmented generation improves factual open-domain question answering.

## Introduction

Open-domain question answering requires models to answer factual questions using broad world knowledge. Closed-book models such as T5 rely only on parametric memory, which can lead to missing facts or hallucinated answers. Retrieval-Augmented Generation (RAG) addresses this by retrieving relevant passages from an external corpus before generating an answer.
Our implementation compares:

| Model | Role |
|---|---|
| T5-base | Closed-book baseline |
| DPR | Retrieval-only baseline |
| RAG-Token | Retrieval-augmented generator with token-level marginalization |
| RAG-Sequence | Retrieval-augmented generator with sequence-level marginalization |

## Chosen Result

We aimed to reproduce the results of **Table 1** for Natural Questions from the original RAG paper. This result compares closed-book and open-book QA models and supports the paper’s main claim: retrieval-augmented generation improves factual open-domain QA.
Because of Google Colab compute and storage limits, our goal was not to exactly match the original scores, but to reproduce the main trend: **RAG variants should outperform T5 and DPR baselines**. Our report states that we specifically reproduced the Table 1 Natural Questions comparison using T5-base, DPR, RAG-Token, and RAG-Sequence. :contentReference[oaicite:1]{index=1}

## GitHub Contents

```text
RAGs-to-Riches/
├── code/
│   └── notebooks/
│       ├── t5_eval.ipynb
│       ├── dpr_eval.ipynb
│       ├── rag_token_train_eval.ipynb
│       ├── rag_sequence_train_eval.ipynb
│       └── final_comparison.ipynb
├── data/
│   └── README.md
├── results/
│   ├── t5_eval_results.csv
│   ├── dpr_results.csv
│   ├── rag_token_results.csv
│   ├── rag_sequence_results.csv
│   └── final_comparison_table.csv
├── poster/
│   └── final_poster.pdf
├── report/
│   └── 135_Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks_2page_report.pdf
├── README.md
├── LICENSE
└── .gitignore
```
## Reproduction Steps

Install the required packages, then run the notebooks in `code/notebooks/` in this order: `t5_eval.ipynb`, `dpr_eval.ipynb`, `rag_token_train_eval.ipynb`, `rag_sequence_train_eval.ipynb`, and `final_comparison.ipynb`.
A GPU runtime is recommended for the RAG notebooks; each notebook saves its output CSV to `results/`, and `final_comparison.ipynb` combines them into the final tables.

## Results/Insights

RAG-Token achieved the best result (**33.8 F1**), followed by RAG-Sequence (**31.44 F1**), while both outperformed T5-base (**8.1 F1**) and DPR (**13.73 F1**).
The main insight is that retrieval improves factual QA, but retrieval quality remains the bottleneck: weak passages usually lead to weak generated answers.

## Conclusion

Overall, RAG models outperformed the closed-book and DPR baselines, showing that retrieval from external sources improves open-domain QA. Our results also demonstrated that retrieval quality is a major bottleneck: weak retrieved passages often lead to weak final answers.

## References

[1] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela, "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," in Advances in Neural Information Processing Systems, vol. 33, 2020.
[2] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih, "Dense Passage Retrieval for Open-Domain Question Answering," in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, 2020.
[3] Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer, "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension," in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 2020.
[4] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov, "Natural Questions: A Benchmark for Question Answering Research," Transactions of the Association for Computational Linguistics, vol. 7, 2019.
[5] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang, "SQuAD: 100,000+ Questions for Machine Comprehension of Text," in Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, 2016.
[6] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu, "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer," Journal of Machine Learning Research, vol. 21, no. 140, 2020.

## Acknowledgements
This project was completed as part of the course work for CS 4782: Introduction to Deep Learning at Cornell University. We are grateful to the course staff for their contribution and support.
