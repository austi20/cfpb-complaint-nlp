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

- Train/test split is **chronological**, not random: train on earlier complaints, test on
  the most recent slice. A random split leaks time-varying vocabulary and inflates scores.
- Baseline: majority-class predictor, reported explicitly so the model has something to beat.
- Model: TF-IDF (word + character n-grams) into linear classifiers, tuned on a validation
  slice. Macro-F1 is the headline metric because the product classes are badly imbalanced.
- Interpretability: top weighted terms per class, and a normalized confusion matrix.
- Themes: NMF topic modeling run per product on the narrative subset.

## Results

_Populated when the run is complete._

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
python src/train.py          # writes metrics and figures/
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
  accuracy — the label taxonomy was also renamed multiple times over the data's date range
  (e.g. three different label strings for what is effectively "credit reporting"), which is
  exactly the kind of near-duplicate class the plan calls for collapsing before modeling.
- Redaction (`XXXX`) removes names, amounts and dates, so any signal that depended on those is gone.
- Company mix shifts over time, so a chronological test split is a harder and more honest
  evaluation than a random one.

## License

MIT
