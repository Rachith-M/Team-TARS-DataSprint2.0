# Team TARS - Data Sprint 2.0, NLP track

Our solution for the Hindi/Hinglish intent classification task: 55 intents plus class 55 for
out-of-scope requests, scored on macro F1 over all 56 classes.

Kaggle team name: Team TARS

Final submission is `submission.csv` in this repo (public 0.83344, private 0.84934).
SHA-256 of the file: `92fc3b6f737e462b5aee3bbe55e42f81e2ec3c8161e2025ee2100059cd46d85b`

Model: `AkshitaS/bhasha-embed-v0` (revision `64b330ff`), a 12-layer BERT initialized from
`google/muril-base-cased` and trained for Hindi, English and romanized Hindi. We fine-tuned the whole
encoder and added a TF-IDF based lexical head on top.
Total parameter count at inference is 256,571,244.


## Pipeline

Every query is scored twice and the logits are blended, 65% from the first pass and 35% from the
second:

1. the text as typed
2. a normalized copy. For romanized or mixed queries we also swap romanized words back to Devanagari
   using a small dictionary built from the training data

Each pass goes through the encoder and a semantic head (768 -> 56), plus a lexical head that sits on
343,162 sparse TF-IDF features (char 2-5 grams and word 1-2 grams over the raw, romanized and phonetic
versions of the text). The lexical head was there in our earlier frozen-encoder version too. The idea is to
give the model direct evidence from exact spellings and short phrases.

Training settings (see `code/sprint/a2_recipe.py` and `code/sprint/depth_ladder.py`):

- AdamW, encoder lr 2e-5 with layer-wise decay 0.9, head lr 1e-3, 10% warmup
- label smoothing 0.1, batch size 32, max 64 tokens, fp16
- 8 epochs on all organizer rows plus extra augmented rows (below)
- trained on an AMD RX 9060 XT with ROCm, roughly 30 minutes

The extra rows are only used for training: augmented copies of about 65% of the rows at weight 0.7
(romanized versions, Hinglish with a few English words swapped in and fillers added, ASR-style dropped
characters), and phrase views plus "opposite" pairs like lights on vs lights off at weight 0.55. We make
sure none of them match any validation text, otherwise the CV numbers would be inflated.

Where things live:

- `code/intent_model.py` - cleaning, Devanagari to Roman transliteration, phonetic folding, TF-IDF features
- `code/sprint/augment.py` - number/filler normalization and the augmentation
- `code/sprint/bhasha_audit.py`, `code/sprint/score_native.py` - the cross-script dictionary view
- `code/sprint/depth_ladder.py` - model, training loop, weight export
- `code/sprint/a2_recipe.py` - validation runs and the final training run that writes the CSV
- `code/sprint/defense.py` and `code/sprint/defense_web/` - CPU demo server and web UI for stage 2
- `code/param_count.py` - parameter count for a trained checkpoint

The rest of `code/sprint/` is support code that the files above import (feature builders, the CPU
runtime, export helpers). We removed our dead-end experiments from this repo so only code on the path to
the CSV and the demo is left.


## Handling Hindi vs Hinglish, and class 55

The biggest thing we noticed in the data is that the test set is about 64% romanized while training is
only 33%. So most of the work went into making the different scripts look the same to the model:

- unicode NFKC, lowercase, remove zero-width joiners and punctuation
- for the TF-IDF features, transliterate Devanagari to Roman and drop the implicit "a" sounds, then fold
  common spelling variations (aa/a, ee/i, w/v, doubled letters)
- convert numbers and drop fillers at the start or end of the query (bhai, yaar, please, na) for the
  second view, e.g. "10 baje" becomes "das baje"
- the dictionary view mentioned above, e.g.
  `kal subah 10 baje meri meeting ki yaad dila do` -> `कल सुबह दस बजे मेरी meeting ki याद दिला दो`
- romanized and code-mixed training copies so the model sees both forms of the same request

Class 55 is just treated as the 56th class, trained on its 249 examples. We don't generate phrase or
opposite-pair rows from class 55, only from the supported intents. The Kaggle CSV is a plain
argmax over the 56 outputs. In the live demo we add a confidence policy on top: confident requests get
routed, borderline ones get a "please rephrase", and low-confidence ones are declined as class 55. The
thresholds were fitted on out-of-fold predictions on organizer data.


## Parameter count

Output of `python code/param_count.py runs/final/final/checkpoint` on the checkpoint that produced
`submission.csv`:

```
Encoder weights              236,965,632
Classifier heads              19,260,136
Neural parameters            256,225,768   (= sum(p.numel()) over the loaded model)
TF-IDF vocabulary entries        343,162
Dictionary entries                 2,314
Conservative total           256,571,244   cap 300,000,000  ->  OK
```

The `sum(p.numel())` number on its own is 256,225,768. We also added the TF-IDF vocabulary and the
dictionary to be safe, since both get loaded at inference. It's a single model, not an ensemble. The
trainer also throws an error if a model would go over 300M (`model_for` in `depth_ladder.py`).


## Validation

We used a fixed 5-fold StratifiedGroupKFold split saved in `code/sprint_results/depth_ladder/plan.npz`.
It's grouped on the phonetic form of each query, so different spellings of the same sentence can't end up
on both sides. Inside each fold there is a smaller early-stopping split to pick the number of epochs,
then we retrain on the full training part of the fold and score the held-out part.

To check how the model copes with messy input, we score three versions of the held-out rows: the
original text, a noisy Hinglish rewrite, and an ASR-style version with dropped characters.

Fold 2 (3,387 held-out organizer rows):

| | macro F1 | accuracy |
|---|---|---|
| original text | 0.8680 | 0.8710 |
| noisy Hinglish | 0.8562 | 0.8606 |
| ASR-style noise | 0.8342 | 0.8379 |

On the same fold, romanized rows alone get 0.8726 and class 55 F1 is 0.863.

Other numbers that shaped our decisions:

- fine-tuning the full encoder vs keeping it frozen, on folds 1 and 2 (both trained on the original rows
  only, no augmentation): +0.029 / +0.036 on clean text and +0.021 / +0.041 on noisy text
- our earlier frozen-encoder model had OOF macro F1 of 0.8678 / 0.8670 on folds 1 and 2 and scored 0.83431
  on the public leaderboard. The fine-tuned model, which is our scored entry, got 0.83344 public and
  0.84934 private
- median per-class F1 on fold 2 is 0.874, with 25 of 56 classes at 0.90 or above. The weak ones are
  cooking technique (0.58), general assistant request (0.69), raise volume (0.71) and greeting (0.73).
  The biggest confusions are between related intents (create vs check calendar, looking up info vs
  general requests, cooking technique vs recipes)

We only ran the full pilot on folds 1 and 2. Fold 1's final refit was cut short by the deadline, so the
table above is fold 2.


## Reproducing submission.csv

You need Python 3.12 and a GPU with at least 8 GB of memory. We trained the final model on an AMD card
with ROCm. The code only uses `torch.cuda`, so an NVIDIA GPU with CUDA should work too, though we
haven't rerun this exact recipe on one.

1. Install a GPU build of PyTorch, then the rest:

   ```
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   pip install -r requirements.txt
   ```

   On Windows with an AMD card, `run_pc.ps1` sets up the ROCm PyTorch build for you.

2. Put the competition `train.csv` and `test.csv` in `code/`. We haven't included them in the repo.
   Expected SHA-256 of train.csv: `db68ca1c1f3072353d72e7447d989b968652852dc24d428a4a91c0f0d9df324a`

3. Download the base model at the exact revision we used:

   ```
   python -c "from huggingface_hub import snapshot_download; print(snapshot_download('AkshitaS/bhasha-embed-v0', revision='64b330ff6703ffebdfc55c412a6a8aaf1b2728c8', allow_patterns=['*.json','*.safetensors','vocab.txt']))"
   ```

4. From inside `code/`, train on everything and predict the test set:

   ```
   python -X utf8 -m sprint.a2_recipe --train train.csv --checkpoint <path from step 3> --plan sprint_results/depth_ladder/plan.npz --output ../runs/final --final-epochs 8 --epochs 8 --lr 2e-5 --test test.csv
   ```

   This writes `runs/final/final/submission.csv` and saves the weights to `runs/final/final/checkpoint/`.
   On Windows, `powershell -ExecutionPolicy Bypass -File run_pc.ps1` does steps 1, 3 and 4 in one go.

5. Check the parameter count from the repo root with `python code/param_count.py runs/final/final/checkpoint`.

If you want to see the CV numbers first, run step 4 with `--folds 2 --epochs 12 --output ../runs/fold2`
instead of the `--final-epochs/--epochs 8/--test` options.

Note: seeds are fixed, but GPU kernels aren't fully deterministic (ROCm especially), so a rerun may
not match our CSV exactly. When we reran these steps from this repo on our machine, 4,901 of the 4,912
predictions matched the committed submission.csv.


## Notes on the rules

- single model, 256.6M parameters including the lookup tables
- trained only on the organizer data. No paid LLM APIs and no external lookups of test queries
- `englishize.py` and the `--englishize` / `--llm-rewrites` options in `a2_recipe.py` are from experiments
  we started after the deadline. They're off by default and weren't used for the submission
- the stage 2 demo (`defense.py`) still loads our earlier frozen-encoder model, the one behind our 0.83431
  public score. We're moving it to the fine-tuned model for the final demo round
- we're not publishing the trained weights. Use the steps above to rebuild them
