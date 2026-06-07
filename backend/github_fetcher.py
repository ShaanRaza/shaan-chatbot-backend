"""
github_fetcher.py — Fetches public GitHub repository data for Shaan Raza.
Run this script once to populate knowledge/github_repos.json.
Usage: python github_fetcher.py
"""

import os
import json
import requests
import base64
import time

GITHUB_USERNAME = "ShaanRaza"
OUTPUT_FILE = "knowledge/github_repos.json"

# Repositories listed in Shaan's resume
KNOWN_REPOS = [
    "Zomato_Dataset_Analysis",
    "Automation-Code-for-RTDMS",
    "EPD-Models-openLCA",
    "PowerBI-Dashboards",
    "FMCG-Customer-Churn-Prediction",
    "Case-Competitions",
]

# Fallback descriptions from resume (used if GitHub API fails or repo is private)
RESUME_DESCRIPTIONS = {
    "Zomato_Dataset_Analysis": {
        "description": "Zomato Data Exploration and Analysis using SQL Server and Pandas.",
        "readme_summary": """Project: Zomato Dataset Analysis
Technologies: SQL Server, Pandas, Python
Purpose: Built a restaurant market analysis model to identify high-demand, low-supply cities and localities for expansion opportunities.
Key Techniques: Window functions, CTEs, and joins to rank localities by restaurant density vs customer demand.
Problem Solved: Helped identify underserved markets in the Zomato dataset where new restaurant openings would have the highest ROI.
Outcome: Produced actionable market intelligence from raw restaurant transaction data.
Lessons Learned: Deep understanding of SQL analytical functions; how to translate business questions into structured query patterns.
Architecture: Data ingestion via Pandas → SQL Server for analytical queries → results exported for visualization.
""",
        "topics": ["sql", "pandas", "data-analysis", "restaurant", "market-analysis"],
        "language": "Python"
    },
    "Automation-Code-for-RTDMS": {
        "description": "Automated web scraping of India's RTDMS (Real-Time Data Monitoring System) portal.",
        "readme_summary": """Project: Automation Code for RTDMS
Technologies: Python, Selenium, BeautifulSoup, Excel (openpyxl)
Purpose: Automate extraction of environmental monitoring data from India's RTDMS portal which has dynamically loaded content.
Problem Solved: The RTDMS portal uses JavaScript-rendered tables that cannot be scraped by simple HTTP requests. Selenium was used to control a headless browser to wait for dynamic content, and BeautifulSoup to parse the rendered HTML.
Architecture: Selenium WebDriver (headless Chrome) → Dynamic page wait → BeautifulSoup HTML parser → Pandas DataFrame → Excel output
Key Challenges: Handling dynamic content loading, site-specific wait conditions, and anti-scraping measures.
Outcome: Extracted state- and category-wise environmental data into structured Excel files, reducing manual effort from hours to minutes.
Future Improvements: Add scheduling via cron, cloud deployment, and automated email delivery of reports.
Lessons Learned: Proficiency in Selenium for dynamic pages, handling race conditions with waits, and reliable HTML parsing strategies.
""",
        "topics": ["selenium", "web-scraping", "automation", "python", "beautifulsoup", "environment"],
        "language": "Python"
    },
    "EPD-Models-openLCA": {
        "description": "LCA models using EXIOBASE and Ecoinvent databases for Environmental Product Declarations.",
        "readme_summary": """Project: EPD Models using openLCA
Technologies: openLCA, EXIOBASE, Ecoinvent, ISO 14025, EN 15804
Purpose: Created Life Cycle Assessment (LCA) models to generate ISO 14025 and EN 15804 compliant Environmental Product Declarations (EPDs) for major industrial clients.
Clients: MSSL Group (Motherson Sumi Systems) and Raymond (textile conglomerate)
Problem Solved: Companies needed certified EPDs to comply with environmental disclosure requirements and reduce supply chain carbon footprint reporting burden.
Architecture: Data collection → openLCA model setup → EXIOBASE/Ecoinvent database linking → Impact assessment → EPD report generation
Key Techniques: Scope 1, 2, and 3 emissions modeling, system boundary definition, cut-off criteria application per ISO standards.
Outcome: Delivered compliant EPD documentation for two major enterprise clients, supporting their sustainability reporting obligations.
Lessons Learned: Deep understanding of LCA methodology, GHG accounting standards, and how to translate complex supply chain data into standardized environmental reports.
""",
        "topics": ["lca", "sustainability", "openLCA", "carbon", "EPD", "environment"],
        "language": "Python"
    },
    "PowerBI-Dashboards": {
        "description": "Interactive Power BI dashboards for business analytics across multiple domains.",
        "readme_summary": """Project: PowerBI Dashboards Collection
Technologies: Power BI, DAX, SQL, Excel
Purpose: A portfolio of interactive Power BI dashboards demonstrating data analysis and visualization capabilities across multiple business domains.
Key Dashboards:
- Scope 1-3 GHG Emissions Tracking Dashboard: Monitors carbon emissions across organizational boundaries with drill-down by category
- Sales & Revenue Analytics Dashboard: KPI tracking with trend analysis and forecasting
- Customer Segmentation Dashboard: RFM-based segmentation with DAX calculations
- Operational KPI Dashboard: AHT, SLA, CSAT, and FRT monitoring for contact center operations
Key Features: DAX measures, calculated columns, drill-through pages, dynamic filtering, cost per ton of CO2 metrics
Certification: Shaan holds the PL-300 (Power BI Data Analyst) certification from Microsoft.
Outcome: Demonstrated ability to translate complex datasets into actionable executive-level dashboards.
Lessons Learned: Advanced DAX pattern design, data model optimization, and UX best practices for business intelligence tools.
""",
        "topics": ["power-bi", "dax", "data-visualization", "business-intelligence", "analytics"],
        "language": "DAX"
    },
    "FMCG-Customer-Churn-Prediction": {
        "description": "ML pipeline for customer churn prediction using RFM analysis, Scikit-Learn, and XGBoost.",
        "readme_summary": """Project: FMCG Customer Churn Prediction
Technologies: Python, NumPy, Pandas, Scikit-Learn, XGBoost, K-Means
Dataset: 10,000+ transaction records from an FMCG (Fast-Moving Consumer Goods) company
Purpose: Predict which customers are likely to churn so the business can intervene with targeted retention strategies.
Methodology:
1. RFM Analysis (Recency, Frequency, Monetary): Engineered customer behavior features
2. Data Preprocessing: Handling missing values, feature scaling, encoding
3. Classification Models: Logistic Regression, Random Forest, XGBoost
4. Customer Segmentation: K-Means clustering to group customers by churn risk and behavior
Performance Metrics: ROC-AUC: 0.92+, F1-Score: 0.85+ (XGBoost model)
Key Insight: High-recency, low-frequency customers showed 3x higher churn probability than engaged regulars.
Architecture: Data Ingestion → EDA → Feature Engineering (RFM) → Model Training → Evaluation → Segmentation
Lessons Learned: Practical application of ML pipeline from raw data to business insights; importance of feature engineering over model complexity for tabular data.
Future Improvements: Real-time scoring API, SHAP explainability, A/B test framework for retention interventions.
""",
        "topics": ["machine-learning", "churn-prediction", "xgboost", "scikit-learn", "rfm", "python"],
        "language": "Python"
    },
    "Case-Competitions": {
        "description": "Case study solutions and competition entries including Jamia Case Challenge (3rd place among 380+ participants).",
        "readme_summary": """Project: Case Competitions Portfolio
Purpose: Repository of business case solutions and competition entries demonstrating strategic thinking and analytical skills.
Key Achievement: Secured 3rd position among 380+ participants in the Jamia Case Challenge for developing a comprehensive business solution for SweepIt (now rebranded as Pronto), a rapidly growing startup in the on-demand services sector.
Case: SweepIt / Pronto — On-Demand Home Services Startup
Problem: Identify growth strategy, competitive positioning, and monetization optimization for a startup competing in a crowded market.
Solution Framework: Market sizing, competitive analysis, unit economics modeling, growth hacking strategy, and technology roadmap.
Tools Used: Excel for financial modeling, PowerPoint for presentation, market research methodologies.
Outcome: Top-3 finish, recognition from industry judges.
Skills Demonstrated: Structured problem-solving, market analysis, financial modeling, strategic communication.
""",
        "topics": ["case-study", "business-analysis", "strategy", "competition"],
        "language": "N/A"
    }
}


def fetch_repo_data(username: str, repo_name: str) -> dict:
    """Fetch repository metadata from GitHub API."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Shaan-AI-Chatbot"
    }

    repo_data = {
        "name": repo_name,
        "full_name": f"{username}/{repo_name}",
        "description": "",
        "readme": "",
        "readme_summary": "",
        "language": "",
        "topics": [],
        "stars": 0,
        "forks": 0,
        "url": f"https://github.com/{username}/{repo_name}",
        "fetched_from_api": False
    }

    try:
        # Fetch repo metadata
        resp = requests.get(
            f"https://api.github.com/repos/{username}/{repo_name}",
            headers=headers,
            timeout=10
        )

        if resp.status_code == 200:
            data = resp.json()
            repo_data["description"] = data.get("description") or ""
            repo_data["language"] = data.get("language") or ""
            repo_data["stars"] = data.get("stargazers_count", 0)
            repo_data["forks"] = data.get("forks_count", 0)
            repo_data["fetched_from_api"] = True
            print(f"  ✓ Fetched metadata for {repo_name}")
        else:
            print(f"  ✗ API returned {resp.status_code} for {repo_name}, using resume fallback")

        time.sleep(0.5)  # Rate limit courtesy

        # Fetch README
        readme_resp = requests.get(
            f"https://api.github.com/repos/{username}/{repo_name}/readme",
            headers=headers,
            timeout=10
        )

        if readme_resp.status_code == 200:
            readme_data = readme_resp.json()
            encoded = readme_data.get("content", "")
            if encoded:
                raw_readme = base64.b64decode(encoded).decode("utf-8", errors="ignore")
                # Truncate very long READMEs
                repo_data["readme"] = raw_readme[:5000]
                print(f"  ✓ Fetched README for {repo_name} ({len(raw_readme)} chars)")
        else:
            print(f"  ✗ No README found for {repo_name}")

        time.sleep(0.5)

        # Fetch topics
        topics_resp = requests.get(
            f"https://api.github.com/repos/{username}/{repo_name}/topics",
            headers={**headers, "Accept": "application/vnd.github.mercy-preview+json"},
            timeout=10
        )
        if topics_resp.status_code == 200:
            repo_data["topics"] = topics_resp.json().get("names", [])

    except Exception as e:
        print(f"  ✗ Error fetching {repo_name}: {e}")

    # Fill in with resume fallback data
    fallback = RESUME_DESCRIPTIONS.get(repo_name, {})
    if not repo_data["description"] and fallback.get("description"):
        repo_data["description"] = fallback["description"]
    if not repo_data["readme"]:
        repo_data["readme_summary"] = fallback.get("readme_summary", "")
    if not repo_data["topics"]:
        repo_data["topics"] = fallback.get("topics", [])
    if not repo_data["language"]:
        repo_data["language"] = fallback.get("language", "")

    # Always ensure readme_summary from fallback is available
    if not repo_data.get("readme_summary"):
        repo_data["readme_summary"] = fallback.get("readme_summary", "")

    return repo_data


def fetch_all_repos(username: str, known_repos: list) -> list:
    """Fetch all repos and combine with known repo data."""
    all_repos = []

    print(f"\nFetching GitHub repo data for: {username}")
    print("=" * 60)

    # Also try to discover public repos via API
    try:
        resp = requests.get(
            f"https://api.github.com/users/{username}/repos?per_page=50&sort=updated",
            headers={"Accept": "application/vnd.github.v3+json", "User-Agent": "Shaan-AI-Chatbot"},
            timeout=10
        )
        if resp.status_code == 200:
            discovered = [r["name"] for r in resp.json()]
            print(f"Discovered {len(discovered)} public repos via API")
            # Add any discovered repos not already in our known list
            for r in discovered:
                if r not in known_repos and r not in ["ShaanRaza"]:
                    known_repos.append(r)
    except Exception as e:
        print(f"Could not fetch repo list: {e}")

    for repo_name in known_repos:
        print(f"\nProcessing: {repo_name}")
        repo = fetch_repo_data(username, repo_name)
        all_repos.append(repo)

    return all_repos


def build_rag_content(repo: dict) -> str:
    """Build searchable RAG content string for a repo."""
    parts = [
        f"GitHub Repository: {repo['name']}",
        f"URL: {repo['url']}",
        f"Description: {repo.get('description', 'N/A')}",
        f"Primary Language: {repo.get('language', 'N/A')}",
        f"Topics: {', '.join(repo.get('topics', [])) or 'N/A'}",
    ]

    if repo.get("readme_summary"):
        parts.append(f"\nDetailed Information:\n{repo['readme_summary']}")
    elif repo.get("readme"):
        # Use first 2000 chars of README
        parts.append(f"\nREADME:\n{repo['readme'][:2000]}")

    return "\n".join(parts)


def main():
    os.makedirs("knowledge", exist_ok=True)

    repos = fetch_all_repos(GITHUB_USERNAME, list(KNOWN_REPOS))

    # Add RAG content to each repo
    for repo in repos:
        repo["rag_content"] = build_rag_content(repo)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(repos, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"✓ Saved {len(repos)} repositories to {OUTPUT_FILE}")
    print(f"  Repos with API data: {sum(1 for r in repos if r.get('fetched_from_api'))}")
    print(f"  Repos with README: {sum(1 for r in repos if r.get('readme'))}")
    print(f"  Repos with fallback data: {sum(1 for r in repos if r.get('readme_summary'))}")


if __name__ == "__main__":
    main()
