## Dashboard & Visual Report Generator

Streamlit application to upload a CSV, explore data, display charts, and generate an HTML/PDF report via Jinja2 + WeasyPrint.

### Live demo
- [data-storytelling-git.streamlit.app](https://data-storytelling-git.streamlit.app/)

### Installation

1) Create a virtual environment (recommended):
```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
```
2) Install dependencies:
```powershell
pip install -r requirements.txt
```

If WeasyPrint causes issues on Windows, make sure MSVC tools are available. Recent wheels (> 61) include the required binaries. See WeasyPrint’s documentation if needed.

### Run the app
```powershell
streamlit run app.py
```

### Usage
- Upload a CSV file (ideally with a date column).
- Choose the analysis period.
- View summary tables and charts.
- Generate the report as HTML or PDF.

### Templates
- Drop your `.j2` files into `templates/`.
- A default template is generated if none is present.

### Structure
- `app.py`: Streamlit app + analysis and rendering logic
- `templates/`: Jinja2 templates
- `assets/report.css`: report styles
- `reports/`: outputs (optional)

### Deploy to Streamlit Community Cloud
- Push is already set up to GitHub: [`michaelgermini/datastorytellinggit`](https://github.com/michaelgermini/datastorytellinggit)
- Steps:
  1) Go to Streamlit Community Cloud and click “New app”.
  2) Select repo: `michaelgermini/datastorytellinggit`
  3) Branch: `main`
  4) Main file path: `app.py`
  5) Deploy.
- Notes:
  - `requirements.txt` is included; build should resolve automatically.
  - PDF generation uses WeasyPrint if native libs are available; otherwise a fallback (xhtml2pdf) is used.
  - A sample CSV is bundled and auto-loaded on startup; you can upload your own CSV from the sidebar.

