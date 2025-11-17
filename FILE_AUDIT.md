# File Audit & Cleanup Plan

## ✅ KEEP - Essential Files

### Core Code (5 files)
1. `fomc_analysis_utils.py` - Core utilities (958 lines)
   - SubtleLinguisticAnalyzer, DiagnosticAnalyzer, ProbabilisticPredictor
2. `run_analysis.py` - Main analysis pipeline (543 lines)
3. `app_streamlit_diagnostic.py` - Production Streamlit app (868 lines) ✅ USES REAL MODELS
4. `parse_fomc_statements.py` - Data acquisition (parses GitHub repo)
5. `run_shap_analysis.py` - SHAP feature importance analysis

### Test Scripts (3 files)
6. `test_diagnostic_system.py` - Tests diagnostic/probabilistic features
7. `test_subtle_features.py` - Tests word-level analyzers
8. `test_different_horizons.py` - Tests 1/2/5/10/20-day predictions

### Visualization (1 file)
9. `create_plots.py` - Generate publication-quality figures (600+ lines)

### Documentation (7 files)
10. `README.md` - Main project documentation
11. `DATA_SOURCES.md` - Where to get FOMC data
12. `DIAGNOSTIC_FLOWCHART.md` - System flowcharts (7 diagrams)
13. `STREAMLIT_README.md` - Streamlit app documentation
14. `CODE_REVIEW.md` - Comprehensive code review
15. `ANALYSIS_SUMMARY.md` - Analysis results
16. `STREAMLIT_INTEGRATION_SUMMARY.md` - Integration summary

### Research Documents (2 files)
17. `RESEARCH_ROADMAP.md` - Publication roadmap
18. `FINE_TUNING_GUIDE.md` - Future improvements
19. `FOMC_Enhanced_Research.ipynb` - Research notebook

### Configuration (2 files)
20. `.gitignore` - Git configuration
21. `requirements_enhanced.txt` - Python dependencies

**Total: 21 files to KEEP**

---

## ❌ DELETE - Obsolete/Redundant Files

### 1. `app_streamlit.py` (600+ lines)
**Reason**: OLD mock Streamlit app
- Uses `np.random.normal(5, 3)` for predictions (line 139)
- Mock feature importance
- Fake similar statements
**Replaced by**: `app_streamlit_diagnostic.py` (real models)

### 2. `download_fomc_data.py` (80 lines)
**Reason**: Alternative download approach using FedTools
- FedTools installation is flaky
- `parse_fomc_statements.py` works better (parses GitHub repo directly)
**Keep instead**: `parse_fomc_statements.py`

### 3. `__pycache__/` (directory)
**Reason**: Python bytecode cache
- Automatically regenerated
- Should be in `.gitignore`

### 4. `*.log` files (2 files)
**Reason**: Generated analysis logs
- `analysis_output.log`
- `run_analysis_output.log`
- Already in `.gitignore`, but exist from previous runs

**Total: 5 items to DELETE**

---

## Summary

- **Keep**: 21 essential files
- **Delete**: 5 obsolete/redundant files
- **Net result**: Clean, focused codebase
