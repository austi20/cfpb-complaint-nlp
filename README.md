# CFPB Complaint Narrative Classifier

Classifying free-text consumer-finance complaint narratives from the CFPB Consumer
Complaint Database into product categories, and surfacing the dominant complaint
themes inside each product with topic modeling.

Every complaint in this dataset is a real person describing, in their own words, a
problem with a bank, credit card, mortgage servicer, debt collector or credit bureau.
The text is messy: inconsistent length, redacted `XXXX` spans, typos, and heavy class
imbalance across products. That is the point — it is the kind of text an analytics team
at a bank or a regulator actually has to route and summarize.

## Questions

1. Can a complaint's product category be predicted from its narrative alone, and by how
   much does that beat guessing the majority class?
2. Which issues does the model systematically confuse, and does that reveal genuine
   overlap in the CFPB taxonomy (for example credit reporting vs. debt collection)?
3. Within each product, what are the recurring complaint themes, and how do they differ
   from the CFPB's own `issue` labels?

## Data

Consumer Financial Protection Bureau, Consumer Complaint Database — originally sourced by CFPB,
pulled here through a historical mirror (see below).

As of 2026-09-16, the live CFPB channels no longer expose narrative text at all: the search
API's `_source` has no narrative field, `&format=csv` on the API has no narrative column, the
bulk `complaints.csv.zip` has no narrative column, and CFPB's own field reference
(`https://cfpb.github.io/api/ccdb/fields.html`) no longer lists one. All three checked and
confirmed narrative-free before falling back to a mirror.

- Data used: [Consumer Complaint Dataset](https://www.kaggle.com/datasets/namigabbasov/consumer-complaint-dataset)
  on Kaggle — a snapshot of the same CFPB database taken while narratives were still public.
  Covers complaints received 2015-03-19 through 2024-07-30. Downloaded via the Kaggle API.
- Field reference (for context on the original source): `https://cfpb.github.io/api/ccdb/fields.html`

Structured fields used: `product`, `sub_product`, `issue`, `company`, `timely`, `state`,
`date_received`. (`company_response` and `sub_issue` aren't in this mirror, so they're
dropped from the original plan's field list.) Text field: `narrative`. The mirror is
pre-filtered to narrative-bearing complaints only, so no separate has-narrative filter is
needed; a working set of 75,000 rows is sampled from the ~2M available for a size that's
fast to iterate on.

## Method

- The split is **chronological**, not random. The 75,000 complaints are sorted by
  `date_received`, the oldest 70% train, the next 15% validate and the newest 15% are held
  out as test. A random split leaks vocabulary that only exists late in the date range and
  inflates the score.
- CFPB renamed its product labels several times, so three spellings of credit reporting and
  three of payday lending are the same product with different strings. Aliases are merged
  first, then any product left with fewer than 500 complaints is collapsed into `Other`
  rather than pretended to be predictable. That leaves 10 classes.
- `XXXX` redaction spans are replaced with a space before vectorizing. They are not signal.
- Baseline: always predict the majority class, reported explicitly so the model has
  something to beat.
- Model: TF-IDF word 1 to 2 grams into a linear classifier, with class weights balanced
  because credit reporting is 60% of the training rows. Macro F1 is the headline metric
  because the classes are badly imbalanced.
- Interpretability: top weighted terms per class, and a confusion matrix normalized by true
  class.
- Themes: NMF topic modeling run per product on the narrative subset.

## Results

The split falls at these dates:

| slice | rows | date range |
|---|---|---|
| train | 52,500 | 2015-03-19 to 2023-06-08 |
| validation | 11,250 | 2023-06-08 to 2024-01-18 |
| test | 11,250 | 2024-01-18 to 2024-07-30 |

Baseline first. Always guessing the majority class, credit reporting, gets 71.4% accuracy on
validation but a macro F1 of **0.083**, which is what happens when one class is 71% of a ten
class problem. Accuracy is the wrong metric here and the baseline is the reason why.

| model | validation accuracy | validation macro F1 |
|---|---|---|
| majority class baseline | 0.714 | 0.083 |
| TF-IDF into LogisticRegression | 0.866 | 0.661 |
| TF-IDF into LinearSVC | 0.879 | **0.668** |

**LinearSVC beats the majority class baseline by 0.585 macro F1 on validation, 0.668 against
0.083.** The two linear models are close enough that the choice barely matters, 0.007 apart.

Two things I checked instead of assuming:

- Character 3 to 5 grams, added alongside the word features, made it slightly worse, 0.662
  against 0.668. They are not in the final model.
- `min_df` is almost flat between 2 and 20, 0.670 down to 0.660. It is set to 5, which keeps
  147,000 features instead of 343,000 for the same score.

![confusion matrix](figures/confusion_matrix.png)

The confusion matrix is normalized by true class, so each row sums to 1 and the small classes
stay readable. Where it goes wrong is the interesting part:

- 26% of debt collection complaints are called credit reporting. That is a genuine overlap,
  not a bug. A consumer disputing a collection account usually describes the credit report
  entry, because the credit report is where they noticed it.
- 31% of money transfer complaints are called bank account. Also reasonable, since a person
  describing a transfer that never arrived is describing their bank.
- Payday or personal loan is the worst class at 0.44 recall and scatters across six others.
  It is the smallest real class and the vocabulary it uses is shared with every other kind of
  loan.

The test slice has not been scored. That happens once, at the end, and becomes the headline
number.

## Repo layout

```
src/            data pull, cleaning, features, modeling
notebooks/      exploratory analysis
figures/        committed charts used in the README
data/raw/       cached API pulls (gitignored)
requirements.txt
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/fetch.py          # caches complaints into data/raw/
python src/train.py          # prints the metrics, writes figures/confusion_matrix.png
```

`fetch.py` downloads from Kaggle, so it needs a Kaggle API token at `~/.kaggle/kaggle.json`
(Kaggle account > Settings > Create New Token). If `data/raw/complaints.csv.zip` is already
cached, it skips the download.

## Caveats

- Data is a historical mirror, not a live pull: it covers complaints received through
  2024-07-30 only, because CFPB stopped publishing narrative text on its live channels
  before this project started (see Data section).
- Narrative-bearing complaints are self-selected: the consumer had to consent to publication,
  so they are not a random sample of complaints.
- `product` labels are assigned at intake and are themselves noisy, which caps achievable
  accuracy. The taxonomy was also renamed several times across the date range, so the aliases
  are merged before modeling (see Method).
- A chronological split means classes can appear or disappear across the cut. `Debt or credit
  management` shows up only after the training cutoff and `Consumer Loan` only before it.
  Both are small enough to land in `Other`, but this is the honest cost of splitting by date,
  and a random split would have hidden it.
- Redaction (`XXXX`) removes names, amounts and dates, so any signal that depended on those is gone.
- Company mix shifts over time, so a chronological test split is a harder and more honest
  evaluation than a random one.

## License

MIT
