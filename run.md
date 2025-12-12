# Quick Start

## Step 1: Clone the repo
```bash
git clone https://github.com/amirraufi/FOMC_Data_Science.git
cd FOMC_Data_Science
```

## Step 2: Create virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

## Step 3: Install dependencies
```bash
pip install -r requirements_enhanced.txt
```

## Step 4: Run analysis
```bash
python run_analysis.py
```

## Step 5: Run utils (generates additional data)
```bash
python fomc_analysis_utils.py
```

## Step 6: Launch the app
```bash
streamlit run app_streamlit_diagnostic.py
```

Open http://localhost:8501 in your browser.
