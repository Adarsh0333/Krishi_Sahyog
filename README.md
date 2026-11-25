# KRISHI Sahyog (कृषि सहयोग) - Digital Farmer Companion & AI Platform

> **Empowering Indian farmers with verified government schemes, localized agro-weather intelligence, fraud protection, and AI-driven crop recommendation.**

---

## 🌾 Platform Architecture Overview

KRISHI Sahyog integrates a mobile-first, high-accessibility web frontend with an AI-driven agricultural intelligence backend:

```
Krishi_Sahyog/
├── frontend/                 # Complete Farmer Web Platform (HTML, CSS, JS, PWA)
│   ├── assets/               # Logos, hero banners & PWA icons
│   ├── css/                  # Typography rules & local font weights
│   ├── data/                 # schemes.json & weather.json datasets
│   ├── js/                   # Modular scripts (schemes, weather, lang, app shell)
│   ├── lang/                 # en.json & hi.json translations (194 keys each)
│   ├── index.html            # Main farmer dashboard
│   ├── schemes.html          # Scheme directory with search & category filters
│   ├── scheme-detail.html    # Detailed criteria, checklists & application steps
│   ├── weather.html          # Agro-meteorological forecasts & crop directives
│   ├── check-scheme.html     # Scam detector & link validator
│   ├── login.html            # Farmer mobile phone authentication
│   ├── otp.html              # 6-digit OTP verification screen
│   ├── settings.html         # Offline cache management & display modes
│   ├── language.html         # First-time language selector
│   ├── manifest.json         # PWA configuration
│   └── service-worker.js     # 100% offline service worker
│
├── krishi-sahyog_AI_part/    # Machine Learning Crop Recommendation Engine
│   ├── dataset/              # Agricultural soil & climate training dataset
│   ├── app.py                # Prediction service application
│   ├── crop_model.pkl        # Trained Machine Learning model
│   ├── requirements.txt.txt  # Python ML dependencies
│   └── train_model.py        # Model training and evaluation script
│
├── designs/                  # Design Sprint Archive & Prototypes
│   ├── screenshots/          # High-resolution screenshots of all pages
│   ├── stitch-exports/       # 14 raw Google Stitch export folders
│   └── README.md             # Design sprint notes & mapping
│
├── scripts/                  # Automated quality & integrity test suite
│   └── audit_check.py        # Automated link, key & asset integrity checker
│
├── .github/workflows/        # Automated GitHub Pages CI/CD pipeline
│   └── deploy.yml            # Auto-deploys frontend/ to GitHub Pages
│
├── start-server.bat          # 1-click local frontend launcher for Windows
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Running the Web Platform Frontend

#### Option A: 1-Click Launch (Windows)
Double-click **`start-server.bat`** in the repository root. It starts a lightweight local server and opens `http://localhost:8000` in your default browser.

#### Option B: Using Python
```bash
python -m http.server -d frontend 8000
```
Open your browser at **`http://localhost:8000`**.

#### Option C: Using Node.js
```bash
cd frontend
npx serve .
```

---

### 2. Running the AI Crop Recommendation Engine

Navigate to the AI backend directory:
```bash
cd krishi-sahyog_AI_part

# Install dependencies
pip install -r requirements.txt.txt

# Run the prediction application
python app.py
```

To re-train the model on updated soil & climate data:
```bash
python train_model.py
```

---

## 🌟 Key Features of the Web Platform

- **Verified Scheme Repository (`schemes.html`, `scheme-detail.html`):** Genuine database of central and state agricultural schemes with clear eligibility criteria, required document checklists, and official links.
- **Scam & Fake Scheme Verifier (`check-scheme.html`):** Real-time client-side search engine and fuzzy matching algorithm to identify predatory WhatsApp links and fake schemes.
- **Localized Agro-Weather (`weather.html`):** District weather updates, dry spell forecasts, and actionable crop directives (e.g., Soybean, Cotton).
- **Single-Language Toggle (`English` / `हिन्दी`):** Dynamic language switching with native Noto Sans Devanagari typography.
- **100% Offline PWA (`service-worker.js`, `manifest.json`):** Works without active internet via Service Worker and cached datasets.
- **Accessibility by Design:** Text sizing (`A- / A / A+`), high-contrast sunlight mode, and Text-to-Speech audio readout for low-literacy farmers.

---

## 🛠️ Verification & Quality Assurance

Run the built-in integrity checker to verify translation key parity and ensure zero broken links:
```bash
python scripts/audit_check.py
```
