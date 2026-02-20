import json
import numpy as np
import pandas as pd


def load_vocab(given_words_path: str = "given_words.json",
               keywords_path: str = "keywords.json"):
    with open(given_words_path) as f:
        given_words = np.array(json.load(f))
    with open(keywords_path) as f:
        keywords = np.array(json.load(f))
    return given_words, keywords


def make_ads_dataset(n: int = 100_000,
                     given_words_path: str = "given_words.json",
                     keywords_path: str = "keywords.json",
                     seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic keyword-ads dataset.

    Parameters
    ----------
    n                 : Number of rows to generate.
    given_words_path  : Path to given_words.json.
    keywords_path     : Path to keywords.json.
    seed              : Random seed.

    Returns
    -------
    DataFrame with columns: given_word, keyword, similarity, competition,
    impressions, clicks, cpc, cost, device, hour, ctr, has_conversion, conversions.
    """
    rng = np.random.default_rng(seed)
    given_words, keywords = load_vocab(given_words_path, keywords_path)

    given = rng.choice(given_words, size=n)
    kw    = rng.choice(keywords, size=n)

    similarity  = rng.uniform(0.05, 0.95, size=n)
    impressions = rng.integers(50, 20000, size=n)
    device      = rng.choice(["mobile", "desktop"], size=n, p=[0.7, 0.3])
    hour        = rng.integers(0, 24, size=n)
    competition = rng.uniform(0.1, 1.0, size=n)

    cpc = np.clip(
        0.2 + 2.0 * competition + 0.5 * (1 - similarity) + rng.normal(0, 0.15, size=n),
        0.05, None
    )

    device_boost = np.where(device == "mobile", 0.02, 0.0)
    hour_boost   = np.where((hour >= 19) & (hour <= 23), 0.01, 0.0)
    ctr = np.clip(
        0.01 + 0.10 * similarity + device_boost + hour_boost + rng.normal(0, 0.01, size=n),
        0.0005, 0.30
    )

    clicks = rng.binomial(impressions, p=ctr)
    cost   = clicks * cpc

    conv_p = 1 / (1 + np.exp(-(-2.0 + 4.0 * similarity - 0.4 * cpc)))
    conversions    = rng.binomial(np.maximum(clicks, 1), p=np.clip(conv_p, 0.0001, 0.8))
    has_conversion = (conversions > 0).astype(int)

    return pd.DataFrame({
        "given_word":     given,
        "keyword":        kw,
        "similarity":     similarity,
        "competition":    competition,
        "impressions":    impressions,
        "clicks":         clicks,
        "cpc":            cpc,
        "cost":           cost,
        "device":         device,
        "hour":           hour,
        "ctr":            np.where(impressions > 0, clicks / impressions, 0.0),
        "has_conversion": has_conversion,
        "conversions":    conversions,
    })
