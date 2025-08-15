## Dashboard & Générateur de Rapports Visuels

Application Streamlit permettant de téléverser un CSV, explorer les données, afficher des graphiques et générer un rapport HTML/PDF via Jinja2 + WeasyPrint.

### Installation

1) Créer un venv (recommandé):
```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
```
2) Installer les dépendances:
```powershell
pip install -r requirements.txt
```

Si WeasyPrint pose problème sous Windows, assurez-vous d’avoir les outils MSVC. Les wheels récents (> 61) incluent les binaires nécessaires. Consultez la documentation WeasyPrint si besoin.

### Lancer l’application
```powershell
streamlit run app.py
```

### Utilisation
- Téléversez un fichier CSV (idéalement avec une colonne date).
- Choisissez la période d’analyse.
- Visualisez les tableaux et graphiques.
- Générez le rapport HTML ou PDF.

### Templates
- Déposez vos `.j2` dans `templates/`.
- Un template par défaut est généré si aucun n’est présent.

### Structure
- `app.py`: app Streamlit + logique d’analyse et de rendu
- `templates/`: templates Jinja2
- `assets/report.css`: styles du rapport
- `reports/`: sorties (si besoin)


