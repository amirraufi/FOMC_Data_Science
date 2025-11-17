"""
FOMC Analysis Utilities
Enhanced functions for analyzing FOMC communications and market reactions

Author: Research Team
Date: 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pandas_datareader.data as web
from difflib import SequenceMatcher
import nltk
from nltk.tokenize import sent_tokenize
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class FOMCDataLoader:
    """Load and prepare FOMC communications and market data"""

    def __init__(self, communications_file='communications.csv'):
        self.communications_file = communications_file
        self.df = None
        self.market_df = None

    def load_communications(self, start_date='2000-01-01'):
        """Load FOMC communications data"""
        print(f"Loading FOMC communications from {self.communications_file}...")

        self.df = pd.read_csv(self.communications_file)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df['Release Date'] = pd.to_datetime(self.df['Release Date'])

        # Filter by date
        self.df = self.df[self.df['Date'] >= start_date].copy()
        self.df = self.df.sort_values('Date').reset_index(drop=True)

        print(f"✓ Loaded {len(self.df)} documents")
        print(f"  Date range: {self.df['Date'].min()} to {self.df['Date'].max()}")

        return self.df

    def fetch_market_data(self, start_date='2000-01-01', end_date=None):
        """
        Fetch market data from FRED

        Returns:
            DataFrame with DFF, DGS2, DGS5, DGS10
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        print(f"Fetching market data from FRED ({start_date} to {end_date})...")

        series = {
            'DFF': 'Effective Federal Funds Rate',
            'DGS2': '2-Year Treasury Yield',
            'DGS5': '5-Year Treasury Yield',
            'DGS10': '10-Year Treasury Yield',
        }

        market_data = {}
        for code, name in series.items():
            try:
                data = web.DataReader(code, 'fred', start_date, end_date)
                market_data[code] = data[code]
                print(f"  ✓ {name}: {len(data)} observations")
            except Exception as e:
                print(f"  ✗ Error fetching {name}: {e}")

        self.market_df = pd.DataFrame(market_data)
        self.market_df = self.market_df.fillna(method='ffill')

        print(f"✓ Market data loaded: {self.market_df.shape}")
        return self.market_df


class MarketReactionCalculator:
    """Calculate market reactions to FOMC events"""

    @staticmethod
    def compute_reactions(df, market_df, horizons=[1, 2]):
        """
        Compute market reactions around FOMC release dates

        Args:
            df: DataFrame with FOMC communications (must have 'Release Date')
            market_df: DataFrame with market data (indexed by date)
            horizons: List of days to compute reactions over

        Returns:
            DataFrame with market reaction columns
        """
        df = df.copy()
        df['Release Date'] = pd.to_datetime(df['Release Date'])

        # Initialize columns
        for horizon in horizons:
            for col in ['DFF', 'DGS2', 'DGS5', 'DGS10']:
                df[f'{col.lower()}_{horizon}d_chg'] = np.nan
                df[f'{col.lower()}_{horizon}d_bp'] = np.nan

        # Compute reactions
        for idx, row in df.iterrows():
            release_date = row['Release Date']

            pre_dates = market_df.index[market_df.index < release_date]
            if len(pre_dates) == 0:
                continue
            pre_date = pre_dates[-1]

            for horizon in horizons:
                target_date = release_date + timedelta(days=horizon)
                post_dates = market_df.index[
                    (market_df.index >= release_date) &
                    (market_df.index <= target_date + timedelta(days=5))
                ]

                if len(post_dates) == 0:
                    continue

                post_date = post_dates[min(horizon-1, len(post_dates)-1)]

                for col in ['DFF', 'DGS2', 'DGS5', 'DGS10']:
                    pre_val = market_df.loc[pre_date, col]
                    post_val = market_df.loc[post_date, col]

                    if pd.notna(pre_val) and pd.notna(post_val):
                        change = post_val - pre_val
                        change_bp = change * 100

                        df.loc[idx, f'{col.lower()}_{horizon}d_chg'] = change
                        df.loc[idx, f'{col.lower()}_{horizon}d_bp'] = change_bp

        # Compute yield curve spreads
        for horizon in horizons:
            df[f'spread_2s10s_{horizon}d_bp'] = (
                df[f'dgs10_{horizon}d_bp'] - df[f'dgs2_{horizon}d_bp']
            )

        print(f"✓ Market reactions computed for {len(df)} releases")
        return df


class ChangeDetector:
    """Detect changes between consecutive FOMC statements"""

    @staticmethod
    def compute_similarity(text1, text2):
        """Compute text similarity using SequenceMatcher"""
        if pd.isna(text1) or pd.isna(text2):
            return np.nan
        return SequenceMatcher(None, text1, text2).ratio()

    @staticmethod
    def extract_key_phrases(text):
        """Extract presence of key policy-related phrases"""
        if pd.isna(text):
            return {}

        text_lower = text.lower()

        return {
            # Inflation
            'inflation_elevated': 'inflation remains elevated' in text_lower or 'elevated inflation' in text_lower,
            'inflation_moderating': 'inflation has moderated' in text_lower or 'moderating inflation' in text_lower,
            'inflation_easing': 'inflation easing' in text_lower or 'inflation has eased' in text_lower,

            # Rates
            'rate_increases': 'rate increase' in text_lower or 'raising the target range' in text_lower,
            'rate_cuts': 'rate cut' in text_lower or 'lowering the target range' in text_lower,
            'rate_hold': 'maintain the target range' in text_lower or 'leaving the target range' in text_lower,

            # Forward guidance
            'data_dependent': 'data dependent' in text_lower or 'incoming data' in text_lower,
            'patient': 'patient' in text_lower and 'policy' in text_lower,
            'gradual': 'gradual' in text_lower,

            # Labor
            'labor_tight': 'tight labor' in text_lower or 'labor market remains tight' in text_lower,
            'labor_softening': 'labor market has softened' in text_lower or 'softening labor' in text_lower,

            # Growth
            'growth_solid': 'solid growth' in text_lower or 'economic growth is solid' in text_lower,
            'growth_slowing': 'slowing growth' in text_lower or 'growth has slowed' in text_lower,
        }

    @staticmethod
    def detect_changes(current_text, previous_text):
        """
        Detect changes between two FOMC statements

        Returns:
            Dictionary of change features
        """
        if pd.isna(current_text) or pd.isna(previous_text):
            return {}

        # Tokenize sentences
        curr_sentences = sent_tokenize(current_text)
        prev_sentences = sent_tokenize(previous_text)

        curr_set = set(s.strip() for s in curr_sentences)
        prev_set = set(s.strip() for s in prev_sentences)

        added = curr_set - prev_set
        removed = prev_set - curr_set
        unchanged = curr_set & prev_set

        # Overall similarity
        overall_similarity = ChangeDetector.compute_similarity(current_text, previous_text)

        # Length changes
        len_change_pct = (len(current_text) - len(previous_text)) / len(previous_text) * 100 if len(previous_text) > 0 else 0
        sentence_count_change = len(curr_sentences) - len(prev_sentences)

        # Key phrase changes
        curr_phrases = ChangeDetector.extract_key_phrases(current_text)
        prev_phrases = ChangeDetector.extract_key_phrases(previous_text)

        phrase_changes = {}
        for phrase_name in curr_phrases.keys():
            curr_val = curr_phrases[phrase_name]
            prev_val = prev_phrases[phrase_name]

            if curr_val and not prev_val:
                phrase_changes[f'{phrase_name}_added'] = 1
            elif not curr_val and prev_val:
                phrase_changes[f'{phrase_name}_removed'] = 1
            else:
                phrase_changes[f'{phrase_name}_added'] = 0
                phrase_changes[f'{phrase_name}_removed'] = 0

        # Compile features
        features = {
            'change_sentences_added': len(added),
            'change_sentences_removed': len(removed),
            'change_sentences_unchanged': len(unchanged),
            'change_net_sentences': len(added) - len(removed),
            'change_pct_sentences_modified': (len(added) + len(removed)) / max(len(prev_set), 1) * 100,
            'change_overall_similarity': overall_similarity,
            'change_text_length_pct': len_change_pct,
            'change_sentence_count': sentence_count_change,
        }

        features.update(phrase_changes)
        return features

    @staticmethod
    def add_change_features(df):
        """
        Add change detection features to DataFrame
        Compares each statement to the previous one

        Now includes BOTH:
        - Sentence-level changes (what we had before)
        - Word-level linguistic features (NEW - the subtle stuff!)
        """
        df = df.copy()
        df = df.sort_values('Date').reset_index(drop=True)

        all_change_features = []

        for idx in range(len(df)):
            if idx == 0:
                all_change_features.append({})
            else:
                current_text = df.loc[idx, 'Text']
                previous_text = df.loc[idx-1, 'Text']

                # Sentence-level changes
                changes = ChangeDetector.detect_changes(current_text, previous_text)

                # Word-level linguistic features (NEW!)
                subtle_features = SubtleLinguisticAnalyzer.analyze_all(current_text, previous_text)

                # Combine both
                changes.update(subtle_features)
                all_change_features.append(changes)

        change_df = pd.DataFrame(all_change_features)
        df = pd.concat([df, change_df], axis=1)

        print(f"✓ Added {len(change_df.columns)} change detection features")
        print(f"  (Includes sentence-level + word-level linguistic features)")
        return df


class SubtleLinguisticAnalyzer:
    """
    Detect WORD-LEVEL linguistic changes between FOMC statements

    Fed language is rigid, so SUBTLE word changes matter:
    - 'transitory' → 'persistent' (policy shift)
    - 'may' → 'will' (certainty change)
    - 'elevated' → 'very elevated' (intensity change)
    - Adding/removing 'not' (negation)
    - Present → Future tense (forward guidance)

    This class captures these subtle shifts that sentence-level analysis misses.
    """

    # Define word lists for different categories
    HEDGE_WORDS = {
        'may', 'might', 'could', 'possibly', 'likely', 'probably',
        'perhaps', 'potentially', 'appears', 'seems', 'suggests'
    }

    CERTAINTY_WORDS = {
        'will', 'shall', 'must', 'certainly', 'definitely',
        'clearly', 'expect', 'expects', 'expected', 'determined'
    }

    NEGATION_WORDS = {
        'not', 'no', 'neither', 'nor', 'never', 'none', 'nobody', 'nothing'
    }

    # Fed-specific word substitutions that signal policy shifts
    FED_WORD_SUBSTITUTIONS = {
        # Inflation timeline
        'transitory': {'intensity': 1, 'category': 'inflation_duration'},
        'temporary': {'intensity': 2, 'category': 'inflation_duration'},
        'persistent': {'intensity': 3, 'category': 'inflation_duration'},

        # Forward guidance
        'patient': {'intensity': 1, 'category': 'policy_stance'},
        'data-dependent': {'intensity': 2, 'category': 'policy_stance'},
        'data dependent': {'intensity': 2, 'category': 'policy_stance'},
        'expeditious': {'intensity': 3, 'category': 'policy_stance'},

        # Rate path speed
        'gradual': {'intensity': 1, 'category': 'rate_path'},
        'measured': {'intensity': 2, 'category': 'rate_path'},
        'rapid': {'intensity': 3, 'category': 'rate_path'},

        # Policy necessity
        'appropriate': {'intensity': 1, 'category': 'policy_urgency'},
        'warranted': {'intensity': 2, 'category': 'policy_urgency'},
        'necessary': {'intensity': 3, 'category': 'policy_urgency'},

        # Inflation descriptors
        'moderating': {'intensity': 1, 'category': 'inflation_level'},
        'elevated': {'intensity': 2, 'category': 'inflation_level'},
        'high': {'intensity': 3, 'category': 'inflation_level'},
        'very high': {'intensity': 4, 'category': 'inflation_level'},
    }

    # Adjective intensifiers
    INTENSIFIERS = {'very', 'highly', 'extremely', 'significantly', 'substantially', 'considerably'}
    DIMINISHERS = {'somewhat', 'slightly', 'moderately', 'relatively', 'fairly'}

    @staticmethod
    def count_word_category(text, word_set):
        """Count occurrences of words from a specific category"""
        if pd.isna(text):
            return 0

        words = text.lower().split()
        return sum(1 for word in words if word in word_set)

    @staticmethod
    def detect_hedge_certainty_changes(current_text, previous_text):
        """
        Track changes in hedge vs certainty language

        More hedging = less certain = potentially dovish
        More certainty = more confident = potentially hawkish
        """
        curr_hedge = SubtleLinguisticAnalyzer.count_word_category(current_text, SubtleLinguisticAnalyzer.HEDGE_WORDS)
        prev_hedge = SubtleLinguisticAnalyzer.count_word_category(previous_text, SubtleLinguisticAnalyzer.HEDGE_WORDS)

        curr_cert = SubtleLinguisticAnalyzer.count_word_category(current_text, SubtleLinguisticAnalyzer.CERTAINTY_WORDS)
        prev_cert = SubtleLinguisticAnalyzer.count_word_category(previous_text, SubtleLinguisticAnalyzer.CERTAINTY_WORDS)

        return {
            'subtle_hedge_word_count_current': curr_hedge,
            'subtle_hedge_word_count_change': curr_hedge - prev_hedge,
            'subtle_certainty_word_count_current': curr_cert,
            'subtle_certainty_word_count_change': curr_cert - prev_cert,
            'subtle_hedge_certainty_ratio': curr_hedge / max(curr_cert, 1),  # Avoid division by zero
        }

    @staticmethod
    def detect_word_substitutions(current_text, previous_text):
        """
        Track Fed-specific word substitutions that signal policy shifts

        Example: 'transitory' → 'persistent' inflation = major shift!
        """
        if pd.isna(current_text) or pd.isna(previous_text):
            return {}

        curr_lower = current_text.lower()
        prev_lower = previous_text.lower()

        features = {}

        # Track each category
        for category in ['inflation_duration', 'policy_stance', 'rate_path', 'policy_urgency', 'inflation_level']:
            curr_intensity = 0
            prev_intensity = 0
            curr_found = False
            prev_found = False

            for word, info in SubtleLinguisticAnalyzer.FED_WORD_SUBSTITUTIONS.items():
                if info['category'] == category:
                    if word in curr_lower:
                        curr_intensity = max(curr_intensity, info['intensity'])
                        curr_found = True
                    if word in prev_lower:
                        prev_intensity = max(prev_intensity, info['intensity'])
                        prev_found = True

            # Calculate intensity change
            if curr_found or prev_found:
                features[f'subtle_{category}_intensity_change'] = curr_intensity - prev_intensity
                features[f'subtle_{category}_intensity_current'] = curr_intensity

        return features

    @staticmethod
    def detect_adjective_intensity_changes(current_text, previous_text):
        """
        Track if adjectives got stronger (very, highly) or weaker (somewhat, slightly)

        'Inflation is elevated' → 'Inflation is very elevated' = hawkish shift
        """
        if pd.isna(current_text) or pd.isna(previous_text):
            return {}

        curr_lower = current_text.lower()
        prev_lower = previous_text.lower()

        curr_intensifiers = SubtleLinguisticAnalyzer.count_word_category(current_text, SubtleLinguisticAnalyzer.INTENSIFIERS)
        prev_intensifiers = SubtleLinguisticAnalyzer.count_word_category(previous_text, SubtleLinguisticAnalyzer.INTENSIFIERS)

        curr_diminishers = SubtleLinguisticAnalyzer.count_word_category(current_text, SubtleLinguisticAnalyzer.DIMINISHERS)
        prev_diminishers = SubtleLinguisticAnalyzer.count_word_category(previous_text, SubtleLinguisticAnalyzer.DIMINISHERS)

        return {
            'subtle_intensifier_count_change': curr_intensifiers - prev_intensifiers,
            'subtle_diminisher_count_change': curr_diminishers - prev_diminishers,
            'subtle_net_intensity_change': (curr_intensifiers - curr_diminishers) - (prev_intensifiers - prev_diminishers),
        }

    @staticmethod
    def detect_negation_changes(current_text, previous_text):
        """
        Track added/removed negations

        'Risks are balanced' → 'Risks are not balanced' = huge meaning flip!
        """
        if pd.isna(current_text) or pd.isna(previous_text):
            return {}

        curr_neg = SubtleLinguisticAnalyzer.count_word_category(current_text, SubtleLinguisticAnalyzer.NEGATION_WORDS)
        prev_neg = SubtleLinguisticAnalyzer.count_word_category(previous_text, SubtleLinguisticAnalyzer.NEGATION_WORDS)

        return {
            'subtle_negation_count_current': curr_neg,
            'subtle_negation_count_change': curr_neg - prev_neg,
        }

    @staticmethod
    def detect_verb_tense_changes(current_text, previous_text):
        """
        Track verb tense shifts (simple rule-based approach)

        'Inflation is elevated' → 'Inflation will ease' = forward guidance change

        Note: This is a simplified version. For production, consider using spaCy POS tagging.
        """
        if pd.isna(current_text) or pd.isna(previous_text):
            return {}

        # Simple indicators for different tenses
        future_indicators = ['will', 'shall', 'going to', 'expect to', 'plan to', 'intend to']
        present_indicators = [' is ', ' are ', ' remains ', ' continues ']
        past_indicators = [' was ', ' were ', ' has ', ' have ', ' had ']

        curr_lower = ' ' + current_text.lower() + ' '
        prev_lower = ' ' + previous_text.lower() + ' '

        curr_future = sum(1 for ind in future_indicators if ind in curr_lower)
        prev_future = sum(1 for ind in future_indicators if ind in prev_lower)

        curr_present = sum(1 for ind in present_indicators if ind in curr_lower)
        prev_present = sum(1 for ind in present_indicators if ind in prev_lower)

        curr_past = sum(1 for ind in past_indicators if ind in curr_lower)
        prev_past = sum(1 for ind in past_indicators if ind in prev_lower)

        return {
            'subtle_future_tense_count_change': curr_future - prev_future,
            'subtle_present_tense_count_change': curr_present - prev_present,
            'subtle_past_tense_count_change': curr_past - prev_past,
            'subtle_future_present_ratio': curr_future / max(curr_present, 1),
        }

    @staticmethod
    def analyze_all(current_text, previous_text):
        """
        Run all word-level linguistic analyses

        Returns:
            Dictionary with all subtle linguistic features
        """
        if pd.isna(current_text) or pd.isna(previous_text):
            return {}

        features = {}

        # 1. Hedge vs Certainty
        features.update(SubtleLinguisticAnalyzer.detect_hedge_certainty_changes(current_text, previous_text))

        # 2. Word Substitutions
        features.update(SubtleLinguisticAnalyzer.detect_word_substitutions(current_text, previous_text))

        # 3. Adjective Intensity
        features.update(SubtleLinguisticAnalyzer.detect_adjective_intensity_changes(current_text, previous_text))

        # 4. Negation
        features.update(SubtleLinguisticAnalyzer.detect_negation_changes(current_text, previous_text))

        # 5. Verb Tense
        features.update(SubtleLinguisticAnalyzer.detect_verb_tense_changes(current_text, previous_text))

        return features


class TimeSeriesSplitter:
    """Create proper time-series train/validation/holdout splits"""

    @staticmethod
    def create_splits(df, holdout_year=2024, cv_cutoff_year=2017):
        """
        Create time-series splits for training

        Args:
            df: DataFrame with FOMC data
            holdout_year: Year to start holdout set (2024)
            cv_cutoff_year: Year to split train/validation (2017)

        Returns:
            Dictionary with train, validation, holdout splits
        """
        df = df.copy()
        df['year'] = pd.to_datetime(df['Date']).dt.year

        train = df[df['year'] < cv_cutoff_year].copy()
        validation = df[(df['year'] >= cv_cutoff_year) & (df['year'] < holdout_year)].copy()
        holdout = df[df['year'] >= holdout_year].copy()

        print(f"Train: {len(train)} samples ({train['year'].min()}-{train['year'].max()})")
        print(f"Validation: {len(validation)} samples ({validation['year'].min() if len(validation) > 0 else 'N/A'}-{validation['year'].max() if len(validation) > 0 else 'N/A'})")
        print(f"Holdout: {len(holdout)} samples ({holdout['year'].min() if len(holdout) > 0 else 'N/A'}-{holdout['year'].max() if len(holdout) > 0 else 'N/A'})")

        return {
            'train': train,
            'validation': validation,
            'holdout': holdout,
            'train_val': pd.concat([train, validation])
        }


class ModelEvaluator:
    """Evaluate models with time-series cross-validation"""

    @staticmethod
    def prepare_features(df, target='dgs2_1d_bp', feature_prefixes=None):
        """
        Prepare feature matrix for modeling

        Args:
            df: DataFrame with all features
            target: Target variable
            feature_prefixes: List of feature name prefixes to include

        Returns:
            X, y, feature_names
        """
        if feature_prefixes is None:
            feature_prefixes = ['change_', 'gpt_', 'bart_', 'finbert_']

        # Select features
        feature_cols = []
        for prefix in feature_prefixes:
            feature_cols.extend([col for col in df.columns if col.startswith(prefix)])

        # Add specific features
        additional_features = ['hawk_minus_dove', 'delta_semantic', 'is_minute']
        for feat in additional_features:
            if feat in df.columns:
                feature_cols.append(feat)

        # Remove duplicates and target-related columns
        feature_cols = list(set(feature_cols))
        feature_cols = [col for col in feature_cols if not any([
            'dgs' in col.lower(),
            'dff' in col.lower(),
            'dy' in col.lower(),
            'spread' in col.lower()
        ])]

        # Extract X and y
        X = df[feature_cols].copy()
        y = df[target].copy()

        # Handle missing values
        X = X.fillna(0)

        # Filter to valid samples
        valid_idx = y.notna()
        X = X[valid_idx]
        y = y[valid_idx]

        print(f"Features: {len(feature_cols)}")
        print(f"Shape: X={X.shape}, y={y.shape}")
        print(f"Target: mean={y.mean():.2f} bp, std={y.std():.2f} bp")

        return X, y, feature_cols


def create_paper_outline():
    """Print academic paper outline"""
    outline = """
    ========================================
    PAPER: "High-Frequency Market Reactions to FOMC Communications: A Multi-Modal NLP Approach"
    ========================================

    1. ABSTRACT
       - Multi-modal NLP analysis of FOMC communications
       - Novel change detection features
       - Predicts high-frequency treasury market reactions

    2. INTRODUCTION
       - Fed communications crucial for policy transmission
       - Markets react within minutes to FOMC releases
       - Research question: Can NLP predict market reactions?

    3. LITERATURE REVIEW
       - FOMC text analysis (Hansen, Lucca & Trebbi)
       - Event studies (Bernanke & Kuttner)
       - NLP in finance

    4. DATA
       - FOMC statements (2000-2025, N=219)
       - Treasury yields, Fed Funds futures
       - Summary statistics

    5. METHODOLOGY
       - 5.1 NLP Features (GPT-4, FinBERT, BART, embeddings)
       - 5.2 NOVEL: Change Detection Features
       - 5.3 Modeling (time-series CV, SHAP)

    6. RESULTS
       - 6.1 Descriptive analysis
       - 6.2 Predictive performance
       - 6.3 Feature importance (SHAP)
       - 6.4 Case studies

    7. ROBUSTNESS CHECKS
       - Alternative specifications
       - Different time periods
       - Statements vs Minutes

    8. DISCUSSION
       - Why change detection matters
       - Limitations
       - Policy implications

    9. CONCLUSION
       - Summary
       - Contributions
       - Future research

    KEY SUCCESS FACTORS:
    ✓ Novel contribution (change detection)
    ✓ Strong empirics (>60% accuracy target)
    ✓ Interpretability (SHAP analysis)
    ✓ Academic rigor (time-series methods)
    ✓ Practical relevance (Fed communication, trading)
    """
    print(outline)


if __name__ == "__main__":
    print("FOMC Analysis Utilities Loaded")
    print("="*60)
    print("\nAvailable classes:")
    print("  - FOMCDataLoader: Load communications and market data")
    print("  - MarketReactionCalculator: Calculate market reactions")
    print("  - ChangeDetector: Detect statement-to-statement changes (sentence-level)")
    print("  - SubtleLinguisticAnalyzer: Detect word-level linguistic changes (NEW!)")
    print("  - TimeSeriesSplitter: Create train/val/holdout splits")
    print("  - ModelEvaluator: Evaluate models with CV")
    print("\nExample usage:")
    print("  from fomc_analysis_utils import FOMCDataLoader, ChangeDetector")
    print("  loader = FOMCDataLoader('communications.csv')")
    print("  df = loader.load_communications()")
    print("\nNEW: Word-level features now included automatically in ChangeDetector!")
