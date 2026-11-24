# Classify & Detoxify

Rewrites toxic comments into polite ones instead of deleting them.

![Demo](https://github.com/user-attachments/assets/89129e03-7253-40ea-aa77-a1a4163d3303)

## How it works

1. **Classify.** RoBERTa, fine-tuned from [`s-nlp/roberta_toxicity_classifier`](https://huggingface.co/s-nlp/roberta_toxicity_classifier), labels each comment as insult, threat, sexual harassment or non-toxic.
2. **Rewrite.** Toxic comments go to LLaMA 3.1 8B, fine-tuned with QLoRA (4-bit, rank 16) through Unsloth, which rewrites them politely and keeps the intent.
3. **Demo.** A multi-user chat room in Gradio.

![Pipeline](https://github.com/user-attachments/assets/a523894a-802b-4a3f-986d-18f5bcf54c12)

## Results

| | |
|---|---|
| Classifier accuracy, held-out 20% | 94.5% |
| Classifier weighted F1 | 0.945 |
| Rewrites, ROUGE-L | 0.30 |

People reading the rewrites preferred ours over baselines with higher ROUGE scores.

## Files

| File | What it does |
|---|---|
| `Classification(Transformer_RoBERTa).ipynb` | Fine-tunes the RoBERTa classifier |
| `DetoxModel(unsloth_Meta_Llama_3_1_8B).ipynb` | Fine-tunes LLaMA 3.1 8B to rewrite toxic comments |
| `DetoxDeploy.ipynb` | Runs the Gradio chat room |
| `toxic_comment_dataset6k.csv` | Training data |

The notebooks were written for Google Colab.

## More

- Case study: [nixoncastroman.github.io/work/classify-detoxify.html](https://nixoncastroman.github.io/work/classify-detoxify.html)
- Write-up: [Medium](https://medium.com/@nixoncastroman08/classify-and-detoxify-tackling-toxic-comments-with-roberta-and-llama-99754e127d94)
- Team: Nixon Castroman ([@XoXonn](https://github.com/XoXonn)), Hans Sanjaya Yantono ([@hansits034](https://github.com/hansits034))
- Builds on [s-nlp/detox](https://github.com/s-nlp/detox)
