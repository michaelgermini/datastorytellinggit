import io
import base64
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from jinja2 import Environment, FileSystemLoader, select_autoescape
"""
WeasyPrint est optionnel sur Windows si les bibliothèques natives (Pango/GObject) manquent.
On l'importera paresseusement pour éviter un crash au démarrage.
"""


APP_DIR = Path(__file__).parent
TEMPLATES_DIR = APP_DIR / "templates"
ASSETS_DIR = APP_DIR / "assets"
REPORTS_DIR = APP_DIR / "reports"
DATA_DIR = APP_DIR / "data"


def ensure_directories_exist() -> None:
    for directory in [TEMPLATES_DIR, ASSETS_DIR, REPORTS_DIR, DATA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    return df


def parse_date_column(df: pd.DataFrame) -> pd.DataFrame:
    # Try detect a date-like column
    date_columns = [
        c for c in df.columns if any(k in c.lower() for k in ["date", "time", "jour", "mois"])
    ]
    if date_columns:
        col = date_columns[0]
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df = df.dropna(subset=[col])
        df = df.sort_values(col)
        df = df.rename(columns={col: "date"})
    else:
        # If no date column exists, create a synthetic monthly period for grouping
        df = df.copy()
        df["date"] = pd.Timestamp.today()
    return df


def filter_by_period(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    mask = (df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))
    return df.loc[mask]


def summarize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.describe(include="all").transpose()
    return summary


def build_sample_dataframe(days: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    start = pd.Timestamp.today().normalize() - pd.Timedelta(days=days)
    dates = pd.date_range(start=start, periods=days, freq="D")

    channels = [
        {"name": "Google Ads", "campaigns": ["Brand", "Generic", "Remarketing"], "ctr": (0.03, 0.08), "cpc": (0.4, 1.2), "cvr": (0.02, 0.05), "aov": (60, 140)},
        {"name": "Facebook Ads", "campaigns": ["Prospection", "Retargeting"], "ctr": (0.01, 0.05), "cpc": (0.2, 0.9), "cvr": (0.01, 0.03), "aov": (40, 120)},
        {"name": "SEO", "campaigns": ["Organic"], "ctr": (0.10, 0.20), "cpc": (0.0, 0.0), "cvr": (0.005, 0.02), "aov": (30, 100)},
    ]

    records = []
    for d in dates:
        k = rng.integers(2, min(4, len(channels)) + 1)
        day_channels = list(rng.choice(channels, size=k, replace=False))
        for ch in day_channels:
            impressions = int(rng.integers(2000, 30001))
            ctr = rng.uniform(*ch["ctr"])
            clicks = max(1, int(round(impressions * ctr)))
            cpc_low, cpc_high = ch["cpc"]
            cpc = rng.uniform(cpc_low, cpc_high) if cpc_high > 0 else 0.0
            spend = round(clicks * cpc, 2)
            cvr = rng.uniform(*ch["cvr"])
            conversions = int(round(clicks * cvr))
            aov = rng.uniform(*ch["aov"])
            revenue = round(conversions * aov, 2)
            sessions = int(clicks * rng.uniform(0.9, 1.3))
            users = int(sessions * rng.uniform(0.7, 0.95))

            records.append(
                {
                    "date": d,
                    "channel": ch["name"],
                    "campaign": rng.choice(ch["campaigns"]).item(),
                    "impressions": impressions,
                    "clicks": clicks,
                    "spend": spend,
                    "revenue": revenue,
                    "conversions": conversions,
                    "sessions": sessions,
                    "users": users,
                }
            )

    return pd.DataFrame.from_records(records)


def get_or_create_sample_csv() -> Path:
    ensure_directories_exist()
    sample_path = DATA_DIR / "sample_marketing.csv"
    if not sample_path.exists():
        df = build_sample_dataframe(days=120)
        df.to_csv(sample_path, index=False)
    return sample_path


def generate_plots(df: pd.DataFrame) -> dict:
    plots = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Line plot over time for the first numeric column, grouped by month
    if "date" in df.columns and numeric_cols:
        metric = numeric_cols[0]
        monthly = (
            df.set_index("date")[metric]
            .resample("MS")
            .sum()
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(monthly["date"], monthly[metric], marker="o")
        ax.set_title(f"{metric} par mois")
        ax.set_xlabel("Mois")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        plots["monthly_line"] = fig

    # Histogram for the first numeric column
    if numeric_cols:
        metric = numeric_cols[0]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(df[metric].dropna(), bins=20, color="#4C78A8")
        ax.set_title(f"Distribution de {metric}")
        ax.set_xlabel(metric)
        ax.set_ylabel("Fréquence")
        ax.grid(True, alpha=0.3)
        plots["histogram"] = fig

    return plots


def fig_to_base64_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def render_html_report(template_name: str, context: dict) -> str:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]) ,
    )
    template = env.get_template(template_name)
    return template.render(**context)


def try_import_weasyprint():
    try:
        from weasyprint import HTML  # type: ignore
        return HTML, None
    except Exception as exc:  # pragma: no cover
        return None, str(exc)


def html_to_pdf_bytes(html_content: str, html_cls=None) -> bytes:
    HTML_cls = html_cls
    if HTML_cls is None:
        HTML_cls, _ = try_import_weasyprint()
    if HTML_cls is None:
        raise RuntimeError("WeasyPrint indisponible: dépendances natives manquantes.")
    pdf = HTML_cls(string=html_content, base_url=str(APP_DIR)).write_pdf()
    return pdf


def try_import_xhtml2pdf():
    try:
        from xhtml2pdf import pisa  # type: ignore
        return pisa, None
    except Exception as exc:  # pragma: no cover
        return None, str(exc)


def html_to_pdf_bytes_fallback(html_content: str) -> bytes:
    # xhtml2pdf fallback (sans dépendances natives). Support CSS limité.
    pisa, err = try_import_xhtml2pdf()
    if pisa is None:
        raise RuntimeError(f"xhtml2pdf indisponible: {err}")
    out = io.BytesIO()
    # xhtml2pdf lit le HTML en bytes; base URL limitée pour les assets
    result = pisa.CreatePDF(src=html_content, dest=out, encoding='utf-8')
    if result.err:
        raise RuntimeError("Echec conversion PDF via xhtml2pdf")
    return out.getvalue()


def main():
    ensure_directories_exist()

    st.set_page_config(page_title="Dashboard & Rapports", layout="wide")
    st.title("Dashboard & Générateur de Rapports Visuels")

    with st.sidebar:
        st.header("Options")
        use_example = st.checkbox("Utiliser un CSV d'exemple", value=True)
        uploaded_file = None
        sample_csv_bytes = None
        sample_path = None
        if use_example:
            sample_path = get_or_create_sample_csv()
            sample_csv_bytes = sample_path.read_bytes()
            st.download_button(
                label="Télécharger le CSV d'exemple",
                data=sample_csv_bytes,
                file_name=sample_path.name,
                mime="text/csv",
            )
        else:
            uploaded_file = st.file_uploader("Téléverser un CSV", type=["csv"]) 

        # Template selection
        available_templates = [p.name for p in TEMPLATES_DIR.glob("*.j2")] or ["rapport_basique.j2"]
        template_choice = st.selectbox("Style du rapport (template)", options=available_templates)

        # Period selection
        today = datetime.today().date()
        default_start = datetime(today.year, today.month, 1)
        default_end = datetime(today.year, today.month, 28)
        start_date = st.date_input("Début", value=default_start)
        end_date = st.date_input("Fin", value=default_end)

    if use_example:
        try:
            df = pd.read_csv(sample_path)
            st.caption(f"CSV d'exemple chargé: {sample_path}")
        except Exception as e:
            st.error(f"Erreur de lecture du CSV d'exemple: {e}")
            return
    else:
        if uploaded_file is None:
            st.info("Veuillez téléverser un fichier CSV pour commencer.")
            return
        try:
            df = load_csv(uploaded_file)
        except Exception as e:
            st.error(f"Erreur de lecture du CSV: {e}")
            return

    df = parse_date_column(df)
    if "date" in df.columns:
        filtered = filter_by_period(df, start=start_date, end=end_date)
    else:
        filtered = df

    st.subheader("Aperçu des données filtrées")
    st.dataframe(filtered.head(100))

    st.subheader("Tableau de synthèse")
    summary = summarize_dataframe(filtered)
    st.dataframe(summary)

    st.subheader("Graphiques")
    plots = generate_plots(filtered)
    col1, col2 = st.columns(2)
    if "monthly_line" in plots:
        with col1:
            st.pyplot(plots["monthly_line"])
    if "histogram" in plots:
        with col2:
            st.pyplot(plots["histogram"])

    # Prepare embedded images for the report
    images_b64 = {k: fig_to_base64_png(v) for k, v in plots.items()}

    # Report metadata and KPIs
    title = "Rapport Mensuel"
    kpis = {}
    numeric_cols = filtered.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        metric = numeric_cols[0]
        kpis = {
            "metric": metric,
            "sum": float(filtered[metric].sum()),
            "mean": float(filtered[metric].mean()),
            "min": float(filtered[metric].min()),
            "max": float(filtered[metric].max()),
        }

    # Default template scaffold if missing
    if template_choice not in [p.name for p in TEMPLATES_DIR.glob("*.j2")]:
        (TEMPLATES_DIR / "rapport_basique.j2").write_text(DEFAULT_TEMPLATE_J2, encoding="utf-8")
        template_choice = "rapport_basique.j2"

    context = {
        "title": title,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "start_date": pd.to_datetime(start_date).date().isoformat(),
        "end_date": pd.to_datetime(end_date).date().isoformat(),
        "kpis": kpis,
        "tables": {
            "summary": summary.reset_index().rename(columns={"index": "colonne"}).to_dict(orient="records"),
        },
        "images": images_b64,
        "stylesheet_path": str(ASSETS_DIR / "report.css"),
    }

    html_content = render_html_report(template_choice, context)

    st.subheader("Génération du rapport")
    colh, colp = st.columns(2)
    with colh:
        st.download_button(
            label="Télécharger HTML",
            data=html_content,
            file_name=f"rapport_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
            mime="text/html",
        )
    with colp:
        # Essayer WeasyPrint d'abord, sinon fallback xhtml2pdf
        HTML_cls, wp_err = try_import_weasyprint()
        pdf_label = "Télécharger PDF"
        pdf_bytes = None
        try:
            if HTML_cls is not None:
                pdf_bytes = html_to_pdf_bytes(html_content, html_cls=HTML_cls)
            else:
                pdf_bytes = html_to_pdf_bytes_fallback(html_content)
                pdf_label = "Télécharger PDF (fallback)"
        except Exception as e:  # pragma: no cover
            st.warning(f"PDF non disponible: {e}")

        if pdf_bytes:
            st.download_button(
                label=pdf_label,
                data=pdf_bytes,
                file_name=f"rapport_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
            )
        elif wp_err:
            st.caption(wp_err)


DEFAULT_TEMPLATE_J2 = """
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{{ title }}</title>
  <link rel="stylesheet" href="{{ stylesheet_path }}" />
  <style>
    /* Fallback minimal CSS if stylesheet missing */
    body { font-family: Arial, sans-serif; margin: 24px; }
    h1, h2 { color: #222; }
    .kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
    .kpi { background: #f4f6f8; padding: 12px; border-radius: 6px; }
    table { border-collapse: collapse; width: 100%; margin-top: 12px; }
    th, td { border: 1px solid #ddd; padding: 6px 8px; font-size: 12px; }
    th { background: #fafafa; text-align: left; }
    img { max-width: 100%; height: auto; }
  </style>
  <meta name="generator" content="Streamlit+Jinja2" />
  <meta name="created" content="{{ generated_at }}" />
</head>
<body>
  <h1>{{ title }}</h1>
  <p>Période: {{ start_date }} → {{ end_date }}</p>
  <h2>Résumé</h2>
  {% if kpis and kpis.metric %}
  <div class="kpis">
    <div class="kpi"><strong>Metric</strong><br/>{{ kpis.metric }}</div>
    <div class="kpi"><strong>Somme</strong><br/>{{ '%.2f'|format(kpis.sum) }}</div>
    <div class="kpi"><strong>Moyenne</strong><br/>{{ '%.2f'|format(kpis.mean) }}</div>
    <div class="kpi"><strong>Min/Max</strong><br/>{{ '%.2f'|format(kpis.min) }} / {{ '%.2f'|format(kpis.max) }}</div>
  </div>
  {% else %}
  <p>Aucun KPI numérique détecté.</p>
  {% endif %}

  <h2>Graphiques</h2>
  {% for name, b64 in images.items() %}
    <div>
      <h3>{{ name }}</h3>
      <img src="data:image/png;base64,{{ b64 }}" alt="{{ name }}" />
    </div>
  {% endfor %}

  <h2>Tableaux</h2>
  {% if tables.summary %}
  <table>
    <thead>
      <tr>
        {% for key in tables.summary[0].keys() %}
        <th>{{ key }}</th>
        {% endfor %}
      </tr>
    </thead>
    <tbody>
      {% for row in tables.summary %}
      <tr>
        {% for value in row.values() %}
        <td>{{ value }}</td>
        {% endfor %}
      </tr>
      {% endfor %}
    </tbody>
  </table>
  {% endif %}

  <footer style="margin-top:24px; font-size:12px; color:#666;">
    Généré le {{ generated_at }}
  </footer>
</body>
</html>
"""


if __name__ == "__main__":
    main()


