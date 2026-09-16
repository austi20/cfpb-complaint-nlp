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

Consumer Financial Protection Bureau, Consumer Complaint Database — public, no API key.

- Search API: `https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/`
  (verified live 2026-09-16; supports `size`, `frm`, `no_aggs`, `field`, `search_term`,
  `date_received_min`, `product`, `has_narrative`)
- Bulk CSV: `https://files.consumerfinance.gov/ccdb/complaints.csv.zip`
- Field reference: `https://cfpb.github.io/api/ccdb/api.html`

Structured fields used: `product`, `sub_product`, `issue`, `sub_issue`, `company`,
`company_response`, `timely`, `state`, `date_received`, `complaint_id`.
Text field: the consumer narrative (`complaint_what_happened` in the API,
`Consumer complaint narrative` in the bulk CSV). Only a minority of complaints carry a
narrative, so the modeling set is the narrative-bearing subset.

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

## Caveats

- Narrative-bearing complaints are self-selected: the consumer had to consent to publication,
  so they are not a random sample of complaints.
- `product` labels are assigned at intake and are themselves noisy, which caps achievable accuracy.
- Redaction (`XXXX`) removes names, amounts and dates, so any signal that depended on those is gone.
- Company mix shifts over time, so a chronological test split is a harder and more honest
  evaluation than a random one.

## License

MIT
