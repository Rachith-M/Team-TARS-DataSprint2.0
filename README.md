# Team TARS · Anveshan Data Sprint 2.0 (NLP track)

Hindi / Hinglish intent classification into 55 assistant intents plus a **Class 55** out-of-scope fallback,
scored by unweighted Macro-F1 over 56 classes.

| | |
|---|---|
| Kaggle team name | `rachith_33` |
| Final submission | [`submission.csv`](submission.csv) (4,912 rows, SHA-256 `92fc3b6f737e462b5aee3bbe55e42f81e2ec3c8161e2025ee2100059cd46d85b`) |
| Leaderboard Macro-F1 | public **0.83344** · private **0.84934** |
| Base model | [`AkshitaS/bhasha-embed-v0`](https://huggingface.co/AkshitaS/bhasha-embed-v0) @ `64b330ff`, a MuRIL-based 12-layer BERT, **fully fine-tuned** |
| Inference parameters | **256,571,244** (≤ 300,000,000), see [Parameter count](#3-parameter-count-proof--300m) |

---

## 1. Pipeline overview & architecture

```
query ─┬─ view 1: original text ──────────────────────┐
       └─ view 2: normalized / cross-script dictionary ┤
                                                       ▼
          Bhasha encoder (12 layers, 768-d, fine-tuned) ── semantic head (768 → 56)
          TF-IDF on raw + romanized + phonetic text ───── lexical head (343,162 → 56)
                                                       ▼
                         56 logits (views averaged) → argmax → intent 0–55
```

* **Encoder.** `bhasha-embed-v0` (MuRIL vocabulary of 197,285 tokens), all 12 layers trained.
* **Lexical head.** A linear layer over 343,162 sparse TF-IDF features: character 2–5-grams and word 1–2-grams
  on three views of the text (raw, romanized, phonetic). It gives exact-spelling evidence the encoder can miss.
* **Training** (`code/sprint/a2_recipe.py`, trainer `code/sprint/depth_ladder.py`): AdamW, encoder LR 2e-5 with
  layer-wise decay 0.9, head LR 1e-3, 10% warmup, label smoothing 0.1, batch 32, max 64 tokens, fp16 AMP,
  8 epochs on all organizer rows plus train-only augmented rows. Trained on an AMD RX 9060 XT (ROCm).
* **Train-only augmentation** (never used for evaluation, blocked from matching any validation text):
  augmented copies of 65% of rows at weight 0.7 (romanized, natural Hinglish with English swaps and fillers,
  ASR-style deletions), plus phrase views and counterfactual contrast pairs (for example lights on ↔ off) at
  weight 0.55.

| File | Role |
|---|---|
| `code/intent_model.py` | Text cleaning, Devanagari → Roman transliteration, phonetic folding, TF-IDF features |
| `code/sprint/augment.py` | Normalization (numbers, fillers) and augmentation |
| `code/sprint/bhasha_audit.py`, `score_native.py` | Cross-script dictionary view |
| `code/sprint/depth_ladder.py` | Model (encoder + heads), trainer, weight export |
| `code/sprint/a2_recipe.py` | Final recipe: fold validation and final training → `submission.csv` |
| `code/sprint/defense.py`, `defense_web/` | Stage 2 CPU demo server and web UI |
| `code/param_count.py` | Parameter-count proof for a trained checkpoint |

## 2. Cross-script (Hindi ↔ Hinglish) & Class 55 handling

**Cross-script.** The test set is 64% romanized against 33% in training, so the pipeline makes every script
look alike:

1. Unicode NFKC, casefold, strip ZWJ/ZWNJ and punctuation (`intent_model.clean`).
2. Devanagari → Roman transliteration with implicit-schwa removal (`intent_model.transliterate`), then phonetic
   folding of spelling variants (aa→a, ee→i, w→v, doubled letters).
3. Numbers and fillers unified (`10 → das`; `bhai`, `yaar`, `please` removed) for the second view.
4. A 2,314-entry cross-script dictionary, built from training rows only, maps romanized words back to
   Devanagari for the second encoder view. Live example:
   `kal subah 10 baje meri meeting ki yaad dila do` → `कल सुबह दस बजे मेरी meeting ki याद दिला दो`.
5. Romanized and code-mixed augmented copies teach the encoder that both scripts mean the same intent.

**Class 55 (out-of-scope).** Class 55 is learned as a 56th class from its 249 organizer examples. The recipe
never generates phrase or contrast rows from Class 55, so the fallback is not diluted by synthetic text.
The Kaggle CSV is the plain argmax over 56 logits. For the live Stage 2 demo, a calibrated policy
(`defense.py`, thresholds fitted on organizer-labeled out-of-fold predictions) routes confident requests,
asks the user to rephrase uncertain ones, and declines the rest as Class 55.

## 3. Parameter count proof (≤ 300M)

`python code/param_count.py runs/final/final/checkpoint` on the checkpoint behind `submission.csv`:

```
Encoder weights              236,965,632
Classifier heads              19,260,136
Neural parameters            256,225,768   (= sum(p.numel()) over the loaded model)
TF-IDF vocabulary entries        343,162
Dictionary entries                 2,314
Conservative total           256,571,244   cap 300,000,000  ->  OK
```

The trainer also refuses to build a model above the cap (`depth_ladder.model_for`). Single model, no ensemble.

## 4. Local CV score & validation summary

Validation uses a frozen 5-fold `StratifiedGroupKFold` plan (`code/sprint_results/depth_ladder/plan.npz`),
grouped by each query's phonetic form and stratified by label × script. Spelling variants of the same query
never straddle train and validation; the loader rejects any plan where they would. Each
pilot fold uses an inner early-stopping split to choose the epoch count, then a refit on the outer
training split. Evaluation views: the organizer text (clean), a noisy Hinglish rewrite and ASR-style noise.

| Held-out pilot fold 2 (3,387 organizer rows) | Macro-F1 | Accuracy |
|---|---|---|
| Clean text | 0.8680 | 0.8710 |
| Noisy Hinglish | 0.8562 | 0.8606 |
| ASR-style noise | 0.8342 | 0.8379 |
| Romanized rows only (clean) | 0.8726 | |
| Class 55 F1 (clean) | 0.863 | |

| Evidence | Result |
|---|---|
| Full fine-tuning vs frozen encoder, pilot folds 1 / 2 (same rows) | +0.029 / +0.036 clean, +0.021 / +0.041 noisy Macro-F1 |
| Frozen-encoder model, clean OOF on folds 1 / 2 | 0.8678 / 0.8670 (public LB 0.83431) |
| Final fine-tuned model, Kaggle | public 0.83344, private 0.84934 |

Per-class F1 on fold 2: median 0.874, 25 of 56 classes ≥ 0.90. The weakest are Cooking technique (0.58),
General assistant request (0.69), Raise volume (0.71) and Greeting (0.73). Most errors fall between neighbouring
intents: create vs check calendar, look-up vs general request, cooking technique vs recipe.

## 5. Reproduce the final submission CSV

Requirements: Python 3.12 and a GPU with ≥ 8 GB memory (NVIDIA CUDA or AMD ROCm; both use `torch.cuda`).
About 30 minutes of training on an RX 9060 XT.

1. **Install PyTorch with GPU support**, then the rest:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu124   # NVIDIA (pick your CUDA)
   pip install -r requirements.txt
   ```
   On Windows with an AMD GPU, `run_pc.ps1` installs the ROCm 7.2.1 PyTorch build automatically.
2. **Add the competition data**: copy the organizers' `train.csv` and `test.csv` into `code/`. They are not
   redistributed here. `train.csv` SHA-256: `db68ca1c1f3072353d72e7447d989b968652852dc24d428a4a91c0f0d9df324a`.
3. **Download the pinned base model**:
   ```bash
   python -c "from huggingface_hub import snapshot_download; print(snapshot_download('AkshitaS/bhasha-embed-v0', revision='64b330ff6703ffebdfc55c412a6a8aaf1b2728c8', allow_patterns=['*.json','*.safetensors','vocab.txt']))"
   ```
4. **Train on all rows and predict the test set** (from `code/`):
   ```bash
   python -X utf8 -m sprint.a2_recipe --train train.csv --checkpoint <path printed in step 3> \
       --plan sprint_results/depth_ladder/plan.npz --output ../runs/final \
       --final-epochs 8 --epochs 8 --lr 2e-5 --test test.csv
   ```
   Output: `runs/final/final/submission.csv`, plus the weights in `runs/final/final/checkpoint/`.
   Windows shortcut: `powershell -ExecutionPolicy Bypass -File run_pc.ps1` runs steps 1, 3 and 4.
5. **Check the parameter count**: `python code/param_count.py runs/final/final/checkpoint`.
6. *(Optional)* Validate on a pilot fold first: replace `--final-epochs 8 --epochs 8 --test test.csv` with
   `--folds 2 --epochs 12` and an output folder such as `../runs/fold2`.

Seeds are fixed, but GPU kernels are not fully deterministic (especially on ROCm), so a rerun can differ
from the committed `submission.csv` on a small number of rows.

## Rules compliance

* One model of 256.6M parameters, under the 300M cap. No ensemble.
* No commercial LLM APIs and no external lookup of test queries; training uses organizer data only.
* `llm_rewrite.py` and `englishize.py` are post-deadline experiments (local, open-weight, training rows only).
  They are **not** part of the submitted pipeline and are off by default.
* Model weights are not published; they can be rebuilt with the steps above.
