# FocusTrack

**Remote Work Productivity Booster** — tracking work hours, breaks and focus,
with adaptive reminders and analytics for remote workers.

Remote workers run long stretches without breaks, lose focus to chat and social
media, and cannot say where the day went. Time trackers only report hours
logged. Fixed timers like Pomodoro interrupt on a schedule that knows nothing
about whether you are deep in something or already drifting.

FocusTrack tracks work hours and breaks automatically, classifies the focus
state every five minutes, sends break reminders and focus nudges only when the
evidence supports them, and turns the result into daily analytics and a Focus
Score — recording **activity counts only, never content**.

This repository is the full implementation of the Milestone 2 report: the
dataset, the eleven-step preprocessing pipeline, both models, the reminder
engine, the Focus Score, the analytics, the live desktop agent and the
dashboard.

---

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -e ".[all]"
focustrack all
```

`focustrack all` simulates the cohort, runs the pipeline, fits both models,
replays the reminder engine and writes everything to `reports/`. It takes about
three minutes. Then:

```bash
focustrack dashboard
```

To smoke-test the whole chain in about twenty seconds:

```bash
focustrack all --quick
```

---

## The five layers

```
  1. Collection  ->  2. Storage  ->  3. Processing  ->  4. Intelligence  ->  5. Interface
  desktop agent      encrypted        the 11-step        focus state +        reminders +
  counts only        local SQLite     engine, every      focus drop          dashboard
                                      5 minutes
```

Every layer runs on the user's own machine. Nothing is uploaded.

| Layer | Package | What it does |
|---|---|---|
| 1 | `focustrack.agent` | Counts keystrokes, clicks, pointer distance, scrolls, window switches and idle seconds once a minute |
| 2 | `focustrack.storage` | Encrypted SQLite; readings are Fernet-encrypted per row, the key never leaves the machine |
| 3 | `focustrack.preprocessing` | The eleven repair-and-feature steps |
| 4 | `focustrack.models`, `focustrack.engine` | The classifier, the drop predictor, the reminder rules and the Focus Score |
| 5 | `focustrack.dashboard` | Streamlit dashboard and desktop notifications |

---

## What is recorded — and what is not

| Recorded, once a minute | Never recorded |
|---|---|
| keystroke **count** | which keys were pressed |
| mouse clicks, pointer distance, scroll events | window or document titles |
| window switches | URLs or page contents |
| idle seconds | screenshots |
| foreground application **name** | network traffic |

The boundary is enforced at the point of collection in
[`agent/collector.py`](src/focustrack/agent/collector.py), not downstream: data
that would need protecting is never captured. `tests/test_models.py` asserts
this, and that stored payloads are not readable without the key.

---

## The dataset

Public remote-work datasets are one-row-per-employee surveys with neither
minute-level activity nor focus labels. So the collection schema is defined
here and a pilot dataset is generated from a documented behaviour model; real
volunteer data is collected against the same schema by the agent.

**40 participants** across five roles (12 developers, 7 each of designers,
writers, analysts and support staff) over **20 workdays** — roughly **382,000
minute records**, **771 user-days** and **37 applications**.

Each minute a participant is in one hidden state — `Focused`, `Distracted`,
`Break` or `Meeting`. Distraction risk rises with minutes since the last break,
again during the 14:00–16:00 slump, and with a personal distractibility trait.

Four overlaps are built in deliberately, because they are what makes the
problem hard and the rule baseline weak:

- **reading looks idle** — focused minutes with almost no keyboard input
- **daydreaming inside work apps** — distracted minutes on a work surface
- **personal chat looks like busy typing** — distraction at full typing speed
- **ambiguous browsing** — distraction in the same browser used for research

| File | Contents |
|---|---|
| `raw_activity_logs.csv` | `user_id, timestamp, active_app, keystrokes, mouse_clicks, mouse_distance_px, scroll_events, window_switches, idle_seconds, state_label` |
| `daily_survey.csv` | End-of-day self-rated productivity (1–10) and energy (1–5); 8% non-response |
| `users.csv` | Role, years of experience, chronotype |

> All data generated here is **simulated**. It shows the pipeline works end to
> end and sets a baseline. Real-world performance is expected to be lower.

---

## The processing engine

The raw log contains the problems a real agent produces, and the pipeline
handles each one. Its guiding rule: **repair what the evidence supports, and
leave the rest empty.**

| # | Step | What it does |
|---|---|---|
| 1 | validate schema | coerce types, drop unparseable timestamps |
| 2 | repair UTC timestamps | detect agents logging in UTC (nobody starts before 06:00 IST) and shift them |
| 3 | drop duplicate uploads | one record per participant per minute |
| 4 | impossible values → missing | stuck keys above 600/min, idle outside 0–60 s, negative counts |
| 5 | fill short gaps | interpolate 1–2 minute stalls; leave longer outages **empty** |
| 6 | protect label integrity | a label is **never** imputed |
| 7 | group apps | 37 applications into 6 categories |
| 8 | detect breaks | three or more consecutive idle minutes |
| 9 | normalise per user | a writer types ~5× faster than a designer |
| 10 | build 5-minute windows | majority state as label; sparse windows dropped |
| 11 | build features | the 31 model inputs |

Steps 4, 5 and 6 are the ones that matter most. Clipping a stuck key to 600
would assert an observation the sensor never made; interpolating a twenty-minute
outage would invent a working stretch; imputing a label would teach the model
the interpolation rule instead of the behaviour.

### The 31 features

| Group | n | Captures |
|---|---|---|
| activity rates | 7 | how hard the keyboard and mouse are working |
| application-usage mix | 8 | which of the six surfaces time went to |
| idle and break signals | 5 | how quiet the window was, and time at the desk |
| time of day | 5 | hour, day progress, the afternoon slump |
| previous 15 minutes | 6 | where the user was coming from |

Every activity rate is the **per-user normalised** count. The statistics come
from a participant's own history, so a new user can be normalised from their own
first days — which is what makes the by-user split honest.

---

## The models

**Focus-state classifier** — predicts the state of each five-minute window.
Four candidates are compared: a rule baseline, logistic regression, a random
forest and histogram gradient boosting.

Selection is on **macro-F1, not accuracy**. `Distracted` is about a tenth of all
windows, so a model that never predicts it still scores near 90% accuracy while
being useless for the one thing the product exists to do. Every learned model is
fitted with balanced class weights for the same reason.

Cross-validation is **grouped by participant**, matching the by-user test split:
ten users — two per role — are held out entirely.

**Focus-drop predictor** — given a user who is focused now, the probability they
will be distracted for five or more of the next fifteen minutes. The honest
control is *time since the last break on its own*, since a fixed timer already
knows that number; both are reported side by side.

---

## The adaptive reminder engine

A break is suggested when:

- the user is focused, the drop probability clears the break threshold, and at
  least **20 minutes** have passed since the last break; **or**
- **100 minutes** have passed at the desk, regardless of the model.

A nudge is sent after **10 continuous minutes** of distraction.

It stays silent during meetings and breaks, waits at least **15 minutes**
between messages, and stops after **6 break reminders** or **4 nudges** a day.
The 100-minute rule fires once and then backs off — a reminder repeated every
fifteen minutes until you comply is the nagging this engine exists to avoid, and
it only trains people to dismiss the notification.

#### A note on the break threshold

The report specifies a threshold of 0.45. Taken literally that number does not
work here, and the reason is worth stating.

The focus-drop model is trained on an event with a ~14% base rate. A
well-calibrated model on a rare event keeps its probabilities low — the 95th
percentile of its predictions is about **0.28**, so a fixed 0.45 is reached by
**0.9%** of windows. Under that setting only **14%** of break reminders came
from the model and the rest fell through to the 100-minute rule: the adaptive
path was effectively dead code, while the headline count still looked healthy.

So the threshold is set by default as a **percentile of the model's own
training distribution** (`drop_threshold_mode: percentile`). That puts **58%**
of reminders on the model and produces **~3.2 break reminders per day** —
matching the volume the report observed. Setting `drop_threshold_mode:
absolute` restores the literal 0.45.

`reports/results.json` records how often the threshold is actually cleared and
which rule fired each reminder, so this is visible rather than buried.

### Focus Score

```
FS = 100 × (0.55 R + 0.25 (1 − S) + 0.20 B)
```

- **R** — focus ratio: focused minutes over minutes *at the desk*. Breaks are
  excluded from the denominator; resting is not a failure to focus.
- **S** — context-switch load, 0–1. Enters as `1 − S`, so fragmented attention
  costs points even when every fragment is work.
- **B** — break balance: full marks when the longest stretch without a break is
  90 minutes or less, tapering to zero at 180. This is the term that separates
  FocusTrack from a stopwatch — **a day of unbroken focus is not a perfect day.**

---

## Commands

```bash
focustrack all [--quick] [--skip-generate]   # the whole study, end to end
focustrack generate [--users N] [--days N]   # simulate the cohort
focustrack preprocess                        # the 11-step pipeline
focustrack train                             # fit both models
focustrack evaluate                          # analytics, comparison, figures
focustrack agent [--minutes N] [--console]   # run the live desktop agent
focustrack report                            # end-of-day self-report
focustrack dashboard [--port 8501]           # open the dashboard
focustrack store [--purge-before DATE]       # inspect the local store
focustrack info                              # config and installed capabilities
```

### Tracking your own work

```bash
focustrack agent          # collects, predicts and reminds; Ctrl+C to stop
focustrack report         # two questions at the end of the day
focustrack store          # what has been recorded
```

The agent degrades gracefully: without `pynput` it cannot count input events and
says so rather than guessing. Install the extras with `pip install -e ".[agent]"`.

---

## Outputs

```
data/raw/          the three dataset files, plus a clean reference copy
data/interim/      the repaired minute-level table
data/processed/    the 5-minute window table, the split and the scaler
models/            both fitted models and their metadata
reports/           results.json, results.md, daily_summary.csv, notifications.csv
reports/figures/   figures 1-5
```

`reports/results.md` is the write-up: the pipeline audit, the model comparison,
the reminder evaluation, the analytics, and a table putting this run's numbers
beside the ones published in the Milestone 2 report. Where they differ, the
difference is shown rather than smoothed over.

| Figure | Shows |
|---|---|
| 1 | the preprocessing pipeline and the effect of each step |
| 2 | the five-layer architecture |
| 3 | confusion matrix on unseen users, and feature importance |
| 4 | one unseen user's day: actual vs predicted, and reminders fired |
| 5 | the daily rhythm and the Focus Score distribution |

Figure 3 shows importance two ways. Per-feature permutation importance splits
credit between correlated features — `keystrokes_mean` and `keystrokes_max`
carry the same signal, so permuting either alone barely hurts. The grouped panel
permutes whole signal families at once and is the better guide to what the model
actually depends on.

---

## Hosting the dashboard

The repository is self-contained: `data/processed/windows.parquet` already
carries `predicted_state` and `drop_probability`, so a deployment needs no
pipeline run and **no model files**. That is deliberate — the fitted random
forest is 122 MB, past GitHub's 100 MB file limit, and a dashboard only needs
the predictions, not the estimator that produced them.

To deploy on [Streamlit Community Cloud](https://share.streamlit.io) (free):

1. Open <https://share.streamlit.io/deploy?repository=anubhav1021%2Ffocustrack&branch=master&mainModule=src%2Ffocustrack%2Fdashboard%2Fapp.py>
2. Sign in with GitHub and authorise Streamlit to read the repository.
3. Confirm the settings — they arrive prefilled:

   | Field | Value |
   |---|---|
   | Repository | `anubhav1021/focustrack` |
   | Branch | `master` |
   | Main file path | `src/focustrack/dashboard/app.py` |
   | Python version | 3.11 – 3.13 |

4. Click **Deploy**. The first build takes a few minutes while the
   dependencies install.

`requirements.txt` is kept free of `pynput` and `plyer` for this reason: they
are desktop-only, and a headless host can neither use nor reliably install
them. The agent extras stay available through `pip install -e ".[agent]"`.

To run the same thing locally:

```bash
focustrack dashboard --port 8507
```

---

## Configuration

Every tunable number lives in [`config.yaml`](config.yaml) — the behaviour
model's hazards, the pipeline's thresholds, the model hyperparameters, the
reminder rules and the Focus Score weights. Nothing is hard-coded in two places.

```bash
focustrack -c my-config.yaml all
```

---

## Development

```bash
pip install -e ".[dev]"
pytest
```

81 tests covering the pipeline's promises (including that it *refuses* to
invent data), the engine's silence rules, the Focus Score's properties, the
drop-label arithmetic, and the store's encryption.

### Layout

```
src/focustrack/
├── config.py, constants.py, io_utils.py
├── data/            personas, behaviour model, log corruption, generation
├── preprocessing/   the 11 steps, breaks, windows, features, split
├── models/          baseline, focus state, focus drop, importance, registry
├── engine/          reminders, focus score, analytics
├── storage/         encrypted SQLite store
├── agent/           collector, notifier, self-report, service loop
├── evaluation/      reminder comparison, figures, the write-up
├── dashboard/       Streamlit app
├── pipeline_runner.py
└── cli.py
```

---

## Limitations

- **The data is simulated.** Scores here are optimistic. The next milestone
  collects two weeks of real data from 10–15 volunteers with self-report labels,
  retrains, and compares.
- **The focus-drop predictor is the weak point.** Predicting a drop fifteen
  minutes ahead is genuinely hard, and it beats time-since-break by a modest
  margin (ROC-AUC 0.63 vs 0.60). It is the main improvement target.
- **Alert recall is bounded by the alert budget, not only by the model.** A
  90-minute timer fires ~200 times across the test users while there are ~1,600
  drops, so no strategy could exceed ~13% recall at that budget. Within it the
  model more than doubles the timer's precision (44% vs 18%). Read the two
  numbers together; recall alone is misleading here.
- **Errors concentrate at transitions.** Accuracy on windows containing a single
  state is far higher than on windows spanning two. A shorter window would cut
  mixed windows but give each one less evidence.
- **A usability study has not been run.** Whether the reminders actually help —
  rather than merely firing at defensible moments — is not something the
  offline evaluation can answer.

---

## References

1. Cirillo, F. (2018). *The Pomodoro Technique.* Currency.
2. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
3. Friedman, J. H. (2001). Greedy function approximation: a gradient boosting
   machine. *Annals of Statistics*, 29(5), 1189–1232.
4. Pedregosa, F., et al. (2011). Scikit-learn: machine learning in Python.
   *JMLR*, 12, 2825–2830.

---

*Milestone 2 — Intelligent Model Design Using AI.
Ahmed Bin Asad (S24CSEU0975), Kunsh Kakkar (S24CSEU1029), Anubhav (S24CSEU1021).*
