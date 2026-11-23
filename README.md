# Transformers Training — DistilBERT Classifier and Phi-2 Alignment

PyTorch/Hugging Face **four-class customer-support text classification** and a separate, unfinished Phi-2 SFT/DPO draft. The Phi-2 draft is not part of the runnable classifier workflow.

> **Implementation status:** `DistilBERT_Colab_Fixed_2026.py` and its Jupyter notebook parse successfully. The classifier has not been run end-to-end as part of this packaging task. `DPO_SFT.py` is preserved **unchanged**.

## Documentation

- [Wiki Home](https://github.com/su394611-del/Transformers-Training/wiki) — published documentation
- [High-Level Design](https://github.com/su394611-del/Transformers-Training/wiki/System-Architecture) — implemented vs planned components
- [Low-Level Design](https://github.com/su394611-del/Transformers-Training/wiki/Low-Level-Design) — functions and library dependencies

The diagrams are available within the published wiki pages; this branch does not contain a separate `diagrams/` directory.

## Main source files

| File | Purpose |
|---|---|
| `DistilBERT_Colab_Fixed_2026.py` | Pretrained DistilBERT, 30 labeled examples, full fine-tuning, frozen-backbone comparison, evaluation and inference |
| `DistilBERT_Colab_Fixed_2026.ipynb` | Jupyter notebook companion |
| `DPO_SFT.py` | Phi-2 LoRA/SFT/DPO draft; do not execute as-is |
| `requirements.txt` | Original unified requirements |
| `requirements_distilbert.txt` | Smaller classifier-only install |

## Run the classifier (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements_distilbert.txt
python DistilBERT_Colab_Fixed_2026.py
```

The first run downloads the pretrained `distilbert-base-uncased` model unless it is cached. The 30-row dataset is embedded in the source and split 24/6 for training/validation. The saved full-model output is placed under `support_classifier/best_model/`. No fixed accuracy figures are claimed from this tiny dataset.

## Editing the Wiki

The published wiki is a separate Git repository at `https://github.com/su394611-del/Transformers-Training.wiki.git`. Edit individual pages using the GitHub Wiki editor, or clone the wiki with Git and push reviewed Markdown changes there. The normal code repository does not contain a `wiki/` directory or a `scripts/publish-wiki.ps1` publisher script.

## Validation

The following checks use the files in the current repository and do not run model training:

```powershell
python -m py_compile DistilBERT_Colab_Fixed_2026.py
python -m json.tool DistilBERT_Colab_Fixed_2026.ipynb > $null
```

These checks verify script syntax and notebook JSON structure only; they do not validate end-to-end training. 
## Privacy and license

Review before publishing; do not commit credentials, confidential examples, cached model weights, `.venv`, or large checkpoints. 

This project is licensed under the Apache License, Version 2.0.© 2026 Sudhanshu Shukla · Licensed under the Apache License, Version 2.0.
