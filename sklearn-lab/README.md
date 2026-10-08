# scikit-learn Lab — interactive, running locally

A dashboard-style lab that runs the scikit-learn workflow
(**Load & split → Fit → Predict → Score & tune**) on a local Python server and
streams each step back to the page.

## What's inside

All charts are interactive **Plotly** charts (hover, zoom, pan, download as PNG).

| View | What it does |
|---|---|
| **Dashboard** | Live pipeline tracker, smart alert (overfitting / healthy), KPI cards (accuracy, F1, ROC AUC, cross-validation; click the corner icon for a plain-English explanation), feature importance / per-class scores / step timing, confusion matrix (counts or row %), ROC curve, PCA prediction map with mistakes marked, confidence histogram, sample predictions, run history |
| **Data Explorer** | Dataset stats, any-two-feature scatter, class-balance donut, per-class distributions, correlation heatmap, first rows |
| **Tuning** | Sweep one hyperparameter; train vs test curves with the generalisation gap shaded |
| **Model Face-off** | All 7 algorithms on the same split: grouped bars, accuracy-vs-speed bubble chart, leaderboard with one-click "Use" |
| **Scenario Lab** | Sliders to invent a row; live prediction with class probabilities |
| **Code & Notes** | Executable 7-line script with terminal output, plus copy-ready code for your last run |

The gear icon opens model settings (hyperparameters, scaling, CV folds). The search box jumps to any view, dataset or model.

**Datasets:** Customer Churn (synthetic), Iris, Wine, Breast Cancer, Digits
**Models:** RandomForest, LogisticRegression, DecisionTree, KNeighbors, SVC, GradientBoosting, GaussianNB

## Run it in VS Code

1. **Install prerequisites (once):** Python 3.10+ from python.org (Windows: tick
   *"Add Python to PATH"*), and the **Python** extension in VS Code
   (Extensions panel → search "Python" by Microsoft).
2. **Open the folder:** unzip `sklearn-lab.zip`, then in VS Code
   *File → Open Folder…* → select `sklearn-lab`.
3. **Open a terminal:** *Terminal → New Terminal* (or `` Ctrl+` ``).
4. **Create a virtual environment:**
   - Windows: `python -m venv .venv`
   - macOS / Linux: `python3 -m venv .venv`
5. **Activate it:**
   - Windows (PowerShell): `.venv\Scripts\Activate.ps1`
     *(if blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then retry)*
   - Windows (cmd): `.venv\Scripts\activate.bat`
   - macOS / Linux: `source .venv/bin/activate`

   You should now see `(.venv)` at the start of the prompt.
6. **Install dependencies:** `pip install -r requirements.txt`
7. **Pick the interpreter:** `Ctrl+Shift+P` → *Python: Select Interpreter* → choose the one in `.venv`.
8. **Start the server** — either:
   - Terminal: `uvicorn app:app --reload --port 8000`
   - or press **F5** (uses `.vscode/launch.json`, opens the browser automatically)
9. **Open** http://127.0.0.1:8000 — the top-right dot turns green when connected.
10. **Stop** with `Ctrl+C` in the terminal (or the red stop button if you used F5).

`--reload` means any edit to `app.py` restarts the server automatically — just refresh the browser.

## Troubleshooting

- **Charts don't appear** → run `pip install -r requirements.txt` again (Plotly is served from the installed `plotly` package, so it works offline).
- **Font looks plain** → Plus Jakarta Sans loads from Google Fonts when you're online; offline it falls back to your system font.
- **"server offline" on the page** → the terminal running uvicorn isn't running or crashed; check it for errors.
- **Port 8000 in use** → use `--port 8001` and open http://127.0.0.1:8001.
- **`uvicorn` not recognised** → the venv isn't active; run step 5 again, or use `python -m uvicorn app:app --reload`.
- **Windows `debugpy` error on F5** → update the VS Code Python extension.

## Extending it

- **Add a model:** add an entry to `MODELS` in `app.py` (class, import line, tunable params, sweep defaults). The UI builds itself from it.
- **Add your own CSV:** add an entry to `DATASETS` whose `load` returns `(X_dataframe, y_series_of_ints, class_names)`.

## API (for reference)

`GET /api/meta` · `GET /api/preview/{dataset}` · `POST /api/run` (NDJSON stream) ·
`POST /api/sweep` (stream) · `POST /api/compare` (stream) · `POST /api/predict` · `POST /api/quickrun`
Interactive API docs: http://127.0.0.1:8000/docs
