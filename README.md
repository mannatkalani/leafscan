# LeafScan — Plant Pathology AI

A Streamlit front end for the CNN built in `CNN_Project.ipynb`: upload a leaf
photo and get back the crop, health status, likely disease, and suggested
next steps — served through a custom dark-green "lab scanner" UI instead of
Streamlit's default theme.

![status](https://img.shields.io/badge/status-demo--ready-3FA34D)

## Features

- Custom dark-forest theme (config.toml + CSS), not Streamlit's default blue
- Animated "scan" sequence on diagnosis (progress readout + scan-line sweep)
- Top-3 prediction confidence shown as styled meters, not a dropped-in chart
- Sidebar system status: model online/offline, param count, class count
- Insights tab: architecture table, optional accuracy curve if you export
  training history
- Graceful "no model loaded" demo mode — the UI is fully browsable even
  before you've exported weights

## Project structure

```
leafscan/
├── app.py                  # Streamlit app (UI + inference)
├── classes.py               # 38 class labels + parsing + care-tip lookup
├── requirements.txt
├── .streamlit/config.toml   # theme (green, dark base)
└── model/
    ├── plant_disease_model.h5   # <- your exported model goes here
    └── history.json             # optional: training history for the accuracy chart
```

## Setup

```bash
pip install -r requirements.txt
```

## Connect your trained model

The notebook trains the model but doesn't export it. Add this as the last
cell after `model.fit(...)`:

```python
import os
os.makedirs("model", exist_ok=True)
model.save("model/plant_disease_model.h5")

# optional, powers the accuracy chart on the Insights tab
import json
with open("model/history.json", "w") as f:
    json.dump(history.history, f)
```

Copy the resulting `model/` folder next to `app.py`. LeafScan checks these
paths in order:

- `model/plant_disease_model.h5`
- `model/plant_disease_model.keras`
- `plant_disease_model.h5`
- `plant_disease_model.keras`

## Run

```bash
streamlit run app.py
```

Without a model file present, the app still runs — the Scan tab shows a
clearly labelled "no model loaded" state instead of crashing, and Insights /
About stay fully usable, so the UI is demoable on its own.

## Notes

- The class order in `classes.py` is the alphabetical folder order Keras'
  `flow_from_directory` assigns automatically — don't reorder it unless you
  also change how the dataset folders are sorted.
- Heads up: the notebook has a Kaggle API token hardcoded in a cell. Rotate
  that key on Kaggle and pull it from an environment variable instead before
  sharing the notebook publicly.
