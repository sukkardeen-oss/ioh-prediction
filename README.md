# IOH Prediction from Intraoperative Arrhythmia

I started this project trying to answer a fairly narrow question. If a patient has an arrhythmia episode during surgery, does that tell you anything about whether they're about to become hypotensive? Anesthesiologists watch for both of these things constantly, but nobody had really tried to model the link between them directly using VitalDB data. That's what this repo is.

The short version of the pipeline: pull 482 patients from VitalDB who have arrhythmia annotations from the PhysioNet VitalDB Arrhythmia Database, clean and merge the clinical and lab data, extract per-episode arrhythmia features, figure out which patients have an invasive arterial line versus just a blood pressure cuff, label each episode for whether hypotension (mean arterial pressure under 65 mmHg) followed within five minutes, then train four different models on the result. 1,284 episodes across 457 patients make it into the final dataset.

## What's actually in here

`notebooks/` has the full data pipeline, numbered 01 through 12 so you can run them in order without guessing what depends on what. There's also an `eda/` folder inside it for the exploratory data; things like the correlation heatmap, the missingness table, and a waveform viewer I leaned on constantly while debugging rhythm labels.

`ml_pipeline/` is the actual training code. Four models: logistic regression, ridge, random forest, and XGBoost. It handles the patient level train and test split so episodes from the same patient never end up on both sides, bootstraps AUROC confidence intervals, and produces SHAP plots for whichever model wins that run.

`analysis/` holds a few scripts that don't produce anything the pipeline depends on, but they shaped real decisions along the way, mostly around whether to trust NIBP estimated blood pressure for patients who didn't have an arterial line.

`data/` splits into `interim`, `processed`, and `final`. If you only care about training models, `data/final/` has what you need and you can skip straight there.

`figures/` has the static analysis figures, the correlation heatmap, the missingness table, and the ECG case examples I pulled for the arrhythmia types.

`outputs/` is empty on purpose except for placeholder folders. Run the pipeline yourself and it fills in with your own trained models and figures. I didn't want old results from some earlier run sitting in the repo forever, quietly going stale while the code around them changed.

`deployment/` is a separate thing entirely. It's a proof of concept FHIR and OMOP interoperability module I put together to sketch out what a hospital deployment might eventually look like. It isn't wired into any of the datasets or models in this repo, and it only works with 12 features. Its own README explains what's actually implemented in it versus what's just described. Treat it as its own project that happens to live in the same place.

## Setup

Two separate environments, because the notebooks and the training pipeline ended up needing different package versions by the time I was done:

```
pip install -r requirements-notebooks.txt   # for notebooks/
pip install -r requirements.txt             # for ml_pipeline/
```

You'll also need to download the VitalDB Arrhythmia Database from PhysioNet before the notebooks will run. `DATA.md` lays out the exact folder it expects and roughly how long each stage takes. The slow part is almost entirely waiting on the VitalDB API, not the actual computation.

## Running the models

Once `data/final/` has what it needs, either from running the notebooks yourself or from what's already committed here, training is one line per dataset:

```
python -m ml_pipeline.main --csv data/final/model_dataset.csv --output_dir outputs/model_dataset
```

Swap in `no_map_dataset.csv`, `no_rhythm_dataset.csv`, `no_map_no_rhythm_dataset.csv`, or `rhythm_only_dataset.csv` for the other four variants. Each run trains all four models, picks the best one by AUROC, and saves ROC curves, a confusion matrix, a results table, and a SHAP summary plot into that output folder.

## Where the numbers landed

Random forest won for three of the five dataset variants. Ridge regression won the rhythm only one. Full model dataset lands around 0.88 AUROC. Take away the rhythm features but keep the MAP derived ones and it barely moves, still sitting around 0.88. Take away MAP but keep rhythm and it drops hard, down to about 0.75. Drop both and you get roughly 0.77, close enough to the MAP removed number that I wouldn't read too much into the exact ordering between those two. Rhythm features alone, with no MAP information at all, only get you to about 0.69.

Even without over-interpreting the small gaps, the pattern holds up. MAP derived features are doing most of the work here. Rhythm characteristics add something real on top of that, they're not nothing.

## A few things worth knowing before you dig in

This is research code, not a production pipeline, and it shows in places. Some of the early data cleaning notebooks reflect decisions made over months as I learned more about the dataset, so don't expect every choice to be obvious just from reading the code. `DATA.md` documents the ones that matter most, including one genuinely debatable call about how episodes with no usable outcome window get labeled. I picked an answer and explained why, but it's a real judgment call, not an obvious default.

I haven't trained or validated anything in `deployment/`. That folder exists to show one possible path from a research model to a clinical interoperability layer, not to claim this model is anywhere near ready for that.
