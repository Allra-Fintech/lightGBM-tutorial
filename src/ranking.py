import pandas as pd
from src.models import FEATURE_COLS, CAT_COLS


def rank_keywords_for_given(given_word: str,
                             candidates: list,
                             base_features: dict,
                             reg,
                             clf) -> pd.DataFrame:
    """Score and rank candidate keywords for a given word.

    Parameters
    ----------
    given_word    : The query / seed word.
    candidates    : List of keyword strings to evaluate.
    base_features : Dict of feature values (all cols except given_word & keyword).
    reg           : Trained LGBMRegressor (CTR model).
    clf           : Trained LGBMClassifier (conversion model).

    Returns
    -------
    DataFrame sorted by score descending.
    """
    rows = [
        {**base_features, "given_word": given_word, "keyword": kw}
        for kw in candidates
    ]
    Xcand = pd.DataFrame(rows)[FEATURE_COLS]
    for c in CAT_COLS:
        Xcand[c] = Xcand[c].astype("category")

    ctr_hat  = reg.predict(Xcand)
    conv_hat = clf.predict_proba(Xcand)[:, 1]

    return pd.DataFrame({
        "given_word":     given_word,
        "keyword":        candidates,
        "pred_ctr":       ctr_hat,
        "pred_conv_prob": conv_hat,
        "score":          ctr_hat * conv_hat,
    }).sort_values("score", ascending=False).reset_index(drop=True)
