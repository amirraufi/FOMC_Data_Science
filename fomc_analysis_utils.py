"""
FOMC Analysis Utilities
Enhanced functions for analyzing FOMC communications and market reactions

Author: Amirhossein Raufi, Edoardo Ponti
Date: 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pandas_datareader.data as web
from difflib import SequenceMatcher
import unicodedata
import re
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
    def _normalize_text(text):
        """
        Normalize text for similarity comparison.

        - Unicode normalization (NFKC)
        - Collapse whitespace
        - Strip leading/trailing whitespace
        """
        if pd.isna(text):
            return ""
        text = unicodedata.normalize("NFKC", str(text))
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def compute_similarity(text1, text2):
        """
        Compute text similarity using token-based SequenceMatcher.

        Uses word tokens instead of raw characters for more meaningful
        similarity scores on FOMC statements (reduces sensitivity to
        whitespace, formatting, and boilerplate).
        """
        if pd.isna(text1) or pd.isna(text2):
            return np.nan

        # Normalize and tokenize
        tokens1 = ChangeDetector._normalize_text(text1).lower().split()
        tokens2 = ChangeDetector._normalize_text(text2).lower().split()

        if not tokens1 or not tokens2:
            return np.nan

        return SequenceMatcher(None, tokens1, tokens2).ratio()

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


class DiagnosticAnalyzer:
    """
    Diagnostic analysis for FOMC statements

    Instead of point predictions, provide:
    1. Percentile scoring (how hawkish vs history)
    2. Change highlighting (what changed linguistically)
    3. Nearest neighbor retrieval (similar past episodes)
    """

    @staticmethod
    def compute_hawkishness_percentile(current_score, historical_scores):
        """
        Score current statement relative to historical distribution

        Args:
            current_score: Current hawkishness score (e.g., gpt_hawk_score)
            historical_scores: Array of historical scores

        Returns:
            Percentile (0-100) where higher = more hawkish
        """
        if pd.isna(current_score):
            return np.nan

        percentile = (historical_scores < current_score).mean() * 100
        return percentile

    @staticmethod
    def create_composite_hawkishness(df):
        """
        Create composite hawkishness score from multiple NLP features

        Combines GPT-4, BART, FinBERT into single score
        """
        scores = []

        # GPT-4 score (normalize to 0-1)
        if 'gpt_hawk_score' in df.columns:
            gpt_norm = (df['gpt_hawk_score'] + 2) / 4  # -2 to +2 → 0 to 1
            scores.append(gpt_norm)

        # BART hawk probability
        if 'bart_hawk_prob' in df.columns:
            scores.append(df['bart_hawk_prob'])

        # FinBERT (positive - negative)
        if 'finbert_pos' in df.columns and 'finbert_neg' in df.columns:
            finbert_score = (df['finbert_pos'] - df['finbert_neg'] + 1) / 2  # -1 to +1 → 0 to 1
            scores.append(finbert_score)

        # Average available scores
        if len(scores) > 0:
            composite = np.mean(scores, axis=0)
        else:
            composite = np.nan

        return composite

    @staticmethod
    def find_nearest_neighbors(current_features, historical_features, k=5, metric='euclidean_standardized'):
        """
        Find k most similar historical statements

        Args:
            current_features: Feature vector for current statement
            historical_features: DataFrame of historical feature vectors
            k: Number of neighbors to return
            metric: 'euclidean_standardized' (recommended), 'cosine', or 'euclidean'

        Returns:
            Indices of k nearest neighbors (most recent first if tied)
        """
        from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
        from sklearn.preprocessing import StandardScaler

        # Convert to numpy arrays
        current = np.array(current_features).reshape(1, -1)
        historical = np.array(historical_features)

        # Compute similarities based on metric
        if metric == 'euclidean_standardized':
            # BEST METHOD: Standardize features first, then use Euclidean distance
            # This ensures all features contribute equally regardless of scale
            all_data = np.vstack([current, historical])
            scaler = StandardScaler()
            all_data_scaled = scaler.fit_transform(all_data)

            current_scaled = all_data_scaled[0:1]
            historical_scaled = all_data_scaled[1:]

            distances = euclidean_distances(current_scaled, historical_scaled)[0]
            # Lower distance = more similar
            neighbor_indices = np.argsort(distances)[:k]
        elif metric == 'cosine':
            similarities = cosine_similarity(current, historical)[0]
            # Higher is more similar
            neighbor_indices = np.argsort(similarities)[::-1][:k]
        else:  # euclidean (raw, not recommended)
            distances = euclidean_distances(current, historical)[0]
            # Lower is more similar
            neighbor_indices = np.argsort(distances)[:k]

        return neighbor_indices

    @staticmethod
    def compute_similarity_score(current_features, historical_features, idx):
        """
        Compute a meaningful similarity score between current and a historical statement.

        Uses standardized Euclidean distance converted to a 0-100% similarity score.
        This gives more meaningful and discriminative similarity values than raw cosine.

        Args:
            current_features: Feature vector for current statement (Series or array)
            historical_features: DataFrame of historical feature vectors
            idx: Index of the historical statement to compare

        Returns:
            Similarity score as percentage (0-100%)
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics.pairwise import euclidean_distances

        # Convert to numpy
        current = np.array(current_features).reshape(1, -1)
        historical_point = np.array(historical_features.iloc[idx]).reshape(1, -1)
        historical_all = np.array(historical_features)

        # Standardize using all historical data as reference
        all_data = np.vstack([current, historical_all])
        scaler = StandardScaler()
        all_data_scaled = scaler.fit_transform(all_data)

        current_scaled = all_data_scaled[0:1]
        historical_scaled = all_data_scaled[1:]
        target_scaled = historical_scaled[idx:idx+1]

        # Compute distance to target
        distance = euclidean_distances(current_scaled, target_scaled)[0][0]

        # Compute all distances to get context for scaling
        all_distances = euclidean_distances(current_scaled, historical_scaled)[0]
        max_distance = np.max(all_distances)

        # Convert distance to similarity (0-100%)
        # Using: similarity = 100 * (1 - distance/max_distance)
        # This gives 100% for identical, 0% for most different
        if max_distance > 0:
            similarity = 100 * (1 - distance / max_distance)
        else:
            similarity = 100.0

        return max(0, min(100, similarity))  # Clamp to 0-100%

    @staticmethod
    def highlight_key_changes(changes_dict):
        """
        Extract most important changes for display

        Args:
            changes_dict: Dictionary from SubtleLinguisticAnalyzer.analyze_all()

        Returns:
            List of human-readable change descriptions
        """
        highlights = []

        # Word substitutions (most important!)
        if changes_dict.get('subtle_inflation_duration_intensity_change', 0) != 0:
            change = changes_dict['subtle_inflation_duration_intensity_change']
            if change > 0:
                highlights.append(f"Inflation language shifted toward 'persistent' (+{change} intensity)")
            else:
                highlights.append(f"Inflation language shifted toward 'transitory' ({change} intensity)")

        if changes_dict.get('subtle_policy_stance_intensity_change', 0) != 0:
            change = changes_dict['subtle_policy_stance_intensity_change']
            if change > 0:
                highlights.append(f"Policy stance shifted toward 'expeditious' (+{change} intensity)")
            else:
                highlights.append(f"Policy stance shifted toward 'patient' ({change} intensity)")

        # Hedge vs certainty
        if changes_dict.get('subtle_hedge_word_count_change', 0) < 0:
            highlights.append(f"Reduced hedging language ({changes_dict['subtle_hedge_word_count_change']} fewer hedge words)")
        elif changes_dict.get('subtle_hedge_word_count_change', 0) > 0:
            highlights.append(f"Increased hedging language (+{changes_dict['subtle_hedge_word_count_change']} hedge words)")

        if changes_dict.get('subtle_certainty_word_count_change', 0) > 0:
            highlights.append(f"Increased certainty language (+{changes_dict['subtle_certainty_word_count_change']} certainty words)")

        # Negation
        if changes_dict.get('subtle_negation_count_change', 0) != 0:
            change = changes_dict['subtle_negation_count_change']
            if change > 0:
                highlights.append(f"Added negations (+{change} 'not'/'no' words - meaning reversal!)")
            else:
                highlights.append(f"Removed negations ({change} fewer 'not'/'no' words)")

        # Intensifiers
        if changes_dict.get('subtle_net_intensity_change', 0) > 0:
            highlights.append(f"Language became more intense (+{changes_dict['subtle_net_intensity_change']} intensifiers)")
        elif changes_dict.get('subtle_net_intensity_change', 0) < 0:
            highlights.append(f"Language became less intense ({changes_dict['subtle_net_intensity_change']} diminishers)")

        # Tense changes
        if changes_dict.get('subtle_future_tense_count_change', 0) > 0:
            highlights.append(f"Increased forward guidance (+{changes_dict['subtle_future_tense_count_change']} future tense)")

        return highlights


class ProbabilisticPredictor:
    """
    Probabilistic predictions instead of point estimates

    Provides:
    1. Conditional distributions (based on similar past episodes)
    2. Quantile predictions (10th, 50th, 90th percentiles)
    3. Tail risk estimates (prob of extreme moves)
    """

    @staticmethod
    def conditional_distribution(current_features, historical_df, feature_cols, target='dy2_1d_bp', k=20):
        """
        Compute conditional distribution based on nearest neighbors

        Args:
            current_features: Feature vector for current statement
            historical_df: DataFrame with features and outcomes
            feature_cols: List of feature column names
            target: Target variable (yield change)
            k: Number of nearest neighbors to use

        Returns:
            Dictionary with quantiles and tail probabilities
        """
        # Find nearest neighbors using standardized Euclidean distance
        historical_features = historical_df[feature_cols].fillna(0)
        neighbor_indices = DiagnosticAnalyzer.find_nearest_neighbors(
            current_features, historical_features, k=k,
            metric='euclidean_standardized'  # Use standardized distance for better similarity
        )

        # Get outcomes for similar statements
        similar_outcomes = historical_df.iloc[neighbor_indices][target].dropna()

        if len(similar_outcomes) == 0:
            return None

        # Compute quantiles
        quantiles = similar_outcomes.quantile([0.1, 0.25, 0.5, 0.75, 0.9])

        # Tail probabilities
        tail_up_10 = (similar_outcomes > 10).mean()
        tail_down_10 = (similar_outcomes < -10).mean()
        prob_positive = (similar_outcomes > 0).mean()

        return {
            'median': quantiles[0.5],
            'q10': quantiles[0.1],
            'q25': quantiles[0.25],
            'q75': quantiles[0.75],
            'q90': quantiles[0.9],
            'mean': similar_outcomes.mean(),
            'std': similar_outcomes.std(),
            'tail_up_10bp': tail_up_10,
            'tail_down_10bp': tail_down_10,
            'prob_positive': prob_positive,
            'n_neighbors': len(similar_outcomes)
        }

    @staticmethod
    def format_probabilistic_forecast(dist_dict, target_name="2Y Treasury"):
        """
        Format conditional distribution as human-readable text

        Args:
            dist_dict: Output from conditional_distribution()
            target_name: Name of target (e.g., "2Y Treasury")

        Returns:
            Formatted string for display
        """
        if dist_dict is None:
            return "Insufficient data for probabilistic forecast"

        forecast = f"""
📊 CONDITIONAL FORECAST - {target_name} (1-day change)

Based on {dist_dict['n_neighbors']} most similar historical statements:

Central Tendency:
  Median outcome: {dist_dict['median']:+.1f} bp
  Mean outcome: {dist_dict['mean']:+.1f} bp (±{dist_dict['std']:.1f} bp std)

Likely Range:
  50% interval: [{dist_dict['q25']:+.1f}, {dist_dict['q75']:+.1f}] bp
  80% interval: [{dist_dict['q10']:+.1f}, {dist_dict['q90']:+.1f}] bp

Directional Probability:
  Prob(yields rise): {dist_dict['prob_positive']:.0%}
  Prob(yields fall): {1-dist_dict['prob_positive']:.0%}

Tail Risks:
  Prob(>+10bp surge): {dist_dict['tail_up_10bp']:.0%}
  Prob(<-10bp drop): {dist_dict['tail_down_10bp']:.0%}
"""
        return forecast


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


class StatementMapPCA:
    """
    PCA-based visualization for FOMC statement exploration.

    Projects the high-dimensional feature space (99 features) to 2D/3D
    for interactive visualization of statement clusters and temporal patterns.

    Use cases:
    - Visualize hawkish vs dovish statement clusters
    - Explore policy regime shifts over 25 years
    - Identify similar statements visually
    - Track Fed language evolution over time
    """

    # Define economic regimes for coloring
    REGIMES = {
        (2000, 2001): ("Dot-com Bust", "#e74c3c"),
        (2002, 2006): ("Recovery & Tightening", "#3498db"),
        (2007, 2009): ("Financial Crisis", "#9b59b6"),
        (2010, 2015): ("Zero Rate Era", "#2ecc71"),
        (2016, 2019): ("Normalization", "#f39c12"),
        (2020, 2021): ("COVID Response", "#1abc9c"),
        (2022, 2025): ("Inflation Fight", "#e74c3c"),
    }

    def __init__(self, n_components=3):
        """
        Initialize PCA for statement mapping.

        Args:
            n_components: Number of PCA components (2 or 3)
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_cols = None
        self.explained_variance_ = None

    def fit_transform(self, df, feature_cols=None):
        """
        Fit PCA on FOMC features and transform to low-dimensional space.

        Args:
            df: DataFrame with FOMC data and features
            feature_cols: List of feature columns to use. If None, auto-detect.

        Returns:
            DataFrame with PCA coordinates (PC1, PC2, PC3) and metadata
        """
        # Auto-detect feature columns if not provided
        if feature_cols is None:
            feature_cols = [col for col in df.columns if (
                col.startswith('change_') or
                col.startswith('subtle_') or
                col.startswith('gpt_') or
                col.startswith('bart_') or
                col.startswith('finbert_') or
                col in ['hawk_cnt', 'dove_cnt', 'hawk_minus_dove', 'cos_prev', 'delta_semantic']
            )]
            # Exclude non-numeric columns
            feature_cols = [col for col in feature_cols
                          if col != 'gpt_reason' and col != 'bart_label'
                          and col in df.columns]

        self.feature_cols = feature_cols

        # Prepare feature matrix
        X = df[feature_cols].fillna(0).values

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Fit and transform PCA
        coords = self.pca.fit_transform(X_scaled)
        self.is_fitted = True
        self.explained_variance_ = self.pca.explained_variance_ratio_

        # Create result DataFrame
        result = pd.DataFrame({
            'PC1': coords[:, 0],
            'PC2': coords[:, 1],
        }, index=df.index)

        if self.n_components >= 3:
            result['PC3'] = coords[:, 2]

        # Add metadata for visualization
        if 'Date' in df.columns:
            result['Date'] = pd.to_datetime(df['Date'])
            result['Year'] = result['Date'].dt.year

        if 'Type' in df.columns:
            result['Type'] = df['Type']

        # Add regime labels
        if 'Date' in df.columns:
            result['Regime'] = result['Year'].apply(self._get_regime_label)
            result['Regime_Color'] = result['Year'].apply(self._get_regime_color)

        # Add hawkishness if available
        hawk_cols = ['gpt_hawk_score', 'composite_hawk']
        for col in hawk_cols:
            if col in df.columns:
                result['Hawkishness'] = df[col]
                break

        # Add market reaction if available
        reaction_cols = ['dy2_1d_bp', 'dgs2_1d_bp']
        for col in reaction_cols:
            if col in df.columns:
                result['Market_Reaction'] = df[col]
                break

        return result

    def transform(self, df):
        """
        Transform new statements using fitted PCA.

        Args:
            df: DataFrame with new statements (must have same feature columns)

        Returns:
            DataFrame with PCA coordinates
        """
        if not self.is_fitted:
            raise ValueError("PCA not fitted. Call fit_transform() first.")

        X = df[self.feature_cols].fillna(0).values
        X_scaled = self.scaler.transform(X)
        coords = self.pca.transform(X_scaled)

        result = pd.DataFrame({
            'PC1': coords[:, 0],
            'PC2': coords[:, 1],
        }, index=df.index)

        if self.n_components >= 3:
            result['PC3'] = coords[:, 2]

        return result

    def _get_regime_label(self, year):
        """Get regime label for a given year."""
        for (start, end), (label, _) in self.REGIMES.items():
            if start <= year <= end:
                return label
        return "Other"

    def _get_regime_color(self, year):
        """Get regime color for a given year."""
        for (start, end), (_, color) in self.REGIMES.items():
            if start <= year <= end:
                return color
        return "#7f8c8d"

    def get_variance_explained(self):
        """
        Get variance explained by each component.

        Returns:
            Dictionary with variance info
        """
        if not self.is_fitted:
            return None

        return {
            'PC1': self.explained_variance_[0],
            'PC2': self.explained_variance_[1],
            'PC3': self.explained_variance_[2] if self.n_components >= 3 else None,
            'total': sum(self.explained_variance_),
            'cumulative': np.cumsum(self.explained_variance_).tolist()
        }

    def get_top_loadings(self, n_top=10):
        """
        Get top feature loadings for each principal component.

        Args:
            n_top: Number of top features to return per component

        Returns:
            Dictionary with top loadings for each PC
        """
        if not self.is_fitted:
            return None

        loadings = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i+1}' for i in range(self.n_components)],
            index=self.feature_cols
        )

        result = {}
        for pc in loadings.columns:
            # Get top positive and negative loadings
            sorted_loadings = loadings[pc].abs().sort_values(ascending=False)
            top_features = sorted_loadings.head(n_top).index.tolist()
            result[pc] = {
                'features': top_features,
                'loadings': loadings.loc[top_features, pc].to_dict()
            }

        return result


if __name__ == "__main__":
    print("FOMC Analysis Utilities Loaded")
    print("="*60)
    print("\nAvailable classes:")
    print("  - FOMCDataLoader: Load communications and market data")
    print("  - MarketReactionCalculator: Calculate market reactions")
    print("  - ChangeDetector: Detect statement-to-statement changes (sentence-level)")
    print("  - SubtleLinguisticAnalyzer: Detect word-level linguistic changes")
    print("  - DiagnosticAnalyzer: Percentile scoring, change highlighting, nearest neighbors")
    print("  - ProbabilisticPredictor: Conditional distributions, quantiles, tail risks")
    print("  - TimeSeriesSplitter: Create train/val/holdout splits")
    print("  - ModelEvaluator: Evaluate models with CV")
    print("  - StatementMapPCA: PCA-based 2D/3D visualization of statement space (NEW!)")
    print("\nExample usage:")
    print("  from fomc_analysis_utils import FOMCDataLoader, StatementMapPCA")
    print("  loader = FOMCDataLoader('communications.csv')")
    print("  df = loader.load_communications()")
    print("  pca_map = StatementMapPCA(n_components=3)")
    print("  coords = pca_map.fit_transform(df)")
    print("\nNEW: PCA Statement Map for visual exploration of Fed language evolution!")
