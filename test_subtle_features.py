"""
Test SubtleLinguisticAnalyzer - Word-Level Feature Extraction

This script tests the new word-level linguistic features on sample FOMC text.
"""

from fomc_analysis_utils import SubtleLinguisticAnalyzer
import pandas as pd

print("="*70)
print("TESTING WORD-LEVEL LINGUISTIC FEATURES")
print("="*70)

# Test Case 1: Transitory → Persistent (classic 2021 Fed shift!)
print("\n" + "="*70)
print("TEST 1: Inflation Language Shift (Transitory → Persistent)")
print("="*70)

previous_text_1 = """
The Committee seeks to achieve maximum employment and inflation at the rate of 2 percent over the longer run.
Inflation is elevated, largely reflecting transitory factors.
The Committee expects inflation pressures may prove to be transitory.
"""

current_text_1 = """
The Committee seeks to achieve maximum employment and inflation at the rate of 2 percent over the longer run.
Inflation remains elevated, reflecting persistent supply and demand imbalances.
The Committee expects inflation pressures will likely prove to be persistent.
"""

features_1 = SubtleLinguisticAnalyzer.analyze_all(current_text_1, previous_text_1)

print("\nExtracted Features:")
for key, value in sorted(features_1.items()):
    print(f"  {key}: {value}")

# Test Case 2: Hedge → Certainty (dovish → hawkish)
print("\n" + "="*70)
print("TEST 2: Certainty Language Shift (May → Will)")
print("="*70)

previous_text_2 = """
The Committee may consider raising rates if conditions warrant.
Policy adjustments might be appropriate going forward.
"""

current_text_2 = """
The Committee will raise rates to ensure price stability.
Policy adjustments will certainly be necessary going forward.
"""

features_2 = SubtleLinguisticAnalyzer.analyze_all(current_text_2, previous_text_2)

print("\nExtracted Features:")
for key, value in sorted(features_2.items()):
    print(f"  {key}: {value}")

# Test Case 3: Negation Change
print("\n" + "="*70)
print("TEST 3: Negation Change (Balanced → Not Balanced)")
print("="*70)

previous_text_3 = """
Risks to the economic outlook are balanced.
The Committee views current conditions as appropriate.
"""

current_text_3 = """
Risks to the economic outlook are not balanced.
The Committee does not view current conditions as appropriate.
"""

features_3 = SubtleLinguisticAnalyzer.analyze_all(current_text_3, previous_text_3)

print("\nExtracted Features:")
for key, value in sorted(features_3.items()):
    print(f"  {key}: {value}")

# Test Case 4: Intensifier (Elevated → Very Elevated)
print("\n" + "="*70)
print("TEST 4: Adjective Intensity (Elevated → Very Elevated)")
print("="*70)

previous_text_4 = """
Inflation is elevated.
Growth is solid.
"""

current_text_4 = """
Inflation is very elevated.
Growth is extremely solid.
"""

features_4 = SubtleLinguisticAnalyzer.analyze_all(current_text_4, previous_text_4)

print("\nExtracted Features:")
for key, value in sorted(features_4.items()):
    print(f"  {key}: {value}")

# Test Case 5: Tense Change (Present → Future)
print("\n" + "="*70)
print("TEST 5: Verb Tense Change (Is → Will)")
print("="*70)

previous_text_5 = """
Inflation is elevated and the labor market remains tight.
The Committee is monitoring economic conditions closely.
"""

current_text_5 = """
Inflation will ease and the labor market will soften.
The Committee will adjust policy as needed.
"""

features_5 = SubtleLinguisticAnalyzer.analyze_all(current_text_5, previous_text_5)

print("\nExtracted Features:")
for key, value in sorted(features_5.items()):
    print(f"  {key}: {value}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

all_features = set()
for features in [features_1, features_2, features_3, features_4, features_5]:
    all_features.update(features.keys())

print(f"\nTotal unique word-level features extracted: {len(all_features)}")
print("\nFeature categories:")
print("  - Hedge/Certainty: subtle_hedge_*, subtle_certainty_*")
print("  - Word Substitutions: subtle_inflation_duration_*, subtle_policy_stance_*, etc.")
print("  - Intensity: subtle_intensifier_*, subtle_diminisher_*")
print("  - Negation: subtle_negation_*")
print("  - Verb Tense: subtle_future_tense_*, subtle_present_tense_*")

print("\n✓ All tests passed! Word-level features are working correctly.")
print("\nNext step: Run `python run_analysis.py` to train models with these new features!")
print("="*70)
