from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path
import random


def generate_sample_csv(output_path: Path, days: int = 120) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "date",
        "channel",
        "campaign",
        "impressions",
        "clicks",
        "spend",
        "revenue",
        "conversions",
        "sessions",
        "users",
    ]

    channels = [
        {
            "name": "Google Ads",
            "campaigns": ["Brand", "Generic", "Remarketing"],
            "ctr": (0.03, 0.08),  # 3-8%
            "cpc": (0.4, 1.2),    # €0.40-€1.20
            "cvr": (0.02, 0.05),  # 2-5%
            "aov": (60, 140),     # €60-€140
        },
        {
            "name": "Facebook Ads",
            "campaigns": ["Prospection", "Retargeting"],
            "ctr": (0.01, 0.05),
            "cpc": (0.2, 0.9),
            "cvr": (0.01, 0.03),
            "aov": (40, 120),
        },
        {
            "name": "SEO",
            "campaigns": ["Organic"],
            "ctr": (0.10, 0.20),  # CTR conceptuellement différent ici
            "cpc": (0.0, 0.0),    # pas de coût direct
            "cvr": (0.005, 0.02),
            "aov": (30, 100),
        },
    ]

    start = date.today() - timedelta(days=days)

    rng = random.Random(42)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for d in (start + timedelta(days=i) for i in range(days)):
            # pour chaque jour, générer entre 2 et 4 lignes (plusieurs canaux)
            day_channels = rng.sample(channels, k=rng.randint(2, min(4, len(channels))))
            for ch in day_channels:
                impressions = rng.randint(2_000, 30_000)
                ctr = rng.uniform(*ch["ctr"])
                clicks = max(1, int(round(impressions * ctr)))
                cpc = rng.uniform(*ch["cpc"]) if ch["cpc"][1] > 0 else 0.0
                spend = round(clicks * cpc, 2)
                cvr = rng.uniform(*ch["cvr"])
                conversions = int(round(clicks * cvr))
                aov = rng.uniform(*ch["aov"])
                revenue = round(conversions * aov, 2)
                sessions = int(clicks * rng.uniform(0.9, 1.3))
                users = int(sessions * rng.uniform(0.7, 0.95))

                writer.writerow(
                    {
                        "date": d.isoformat(),
                        "channel": ch["name"],
                        "campaign": rng.choice(ch["campaigns"]),
                        "impressions": impressions,
                        "clicks": clicks,
                        "spend": spend,
                        "revenue": revenue,
                        "conversions": conversions,
                        "sessions": sessions,
                        "users": users,
                    }
                )


if __name__ == "__main__":
    generate_sample_csv(Path("data/sample_marketing.csv"), days=120)


