import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

FEATURE_COLS = [
    "given_word", "keyword", "similarity", "competition",
    "impressions", "clicks", "cpc", "cost", "device", "hour",
]
CAT_COLS = ["given_word", "keyword", "device"]


def prepare_features(df: pd.DataFrame):
    """Split df into train/test feature matrices and target arrays.

    Returns
    -------
    X_train, X_test, y_ctr_train, y_ctr_test, y_conv_train, y_conv_test
    """
    X = df[FEATURE_COLS].copy()
    for c in CAT_COLS:
        X[c] = X[c].astype("category")

    y_ctr  = df["ctr"].values
    y_conv = df["has_conversion"].values

    X_train, X_test, y_ctr_train, y_ctr_test = train_test_split(
        X, y_ctr, test_size=0.2, random_state=42
    )
    y_conv_train = y_conv[X_train.index]
    y_conv_test  = y_conv[X_test.index]

    return X_train, X_test, y_ctr_train, y_ctr_test, y_conv_train, y_conv_test


def train_ctr_model(X_train, X_test, y_ctr_train, y_ctr_test) -> lgb.LGBMRegressor:
    """Train an impression-weighted CTR regression model."""
    reg = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    reg.fit(
        X_train, y_ctr_train,
        sample_weight=X_train["impressions"],
        eval_set=[(X_test, y_ctr_test)],
        eval_sample_weight=[X_test["impressions"]],
        eval_metric="l2",
        categorical_feature=CAT_COLS,
        callbacks=[
            lgb.early_stopping(stopping_rounds=80, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )
    pred = reg.predict(X_test)
    rmse = mean_squared_error(y_ctr_test, pred, squared=False)
    print(f"CTR RMSE : {rmse:.6f}  |  best iter: {reg.best_iteration_}")
    return reg


def train_conversion_model(X_train, X_test, y_conv_train, y_conv_test) -> lgb.LGBMClassifier:
    """Train a binary conversion classifier."""
    clf = lgb.LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    clf.fit(
        X_train, y_conv_train,
        eval_set=[(X_test, y_conv_test)],
        eval_metric="auc",
        categorical_feature=CAT_COLS,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )
    proba = clf.predict_proba(X_test)[:, 1]
    print(f"AUC   : {roc_auc_score(y_conv_test, proba):.4f}")
    print(f"PR-AUC: {average_precision_score(y_conv_test, proba):.4f}")
    return clf


def train_ranker(df: pd.DataFrame) -> lgb.LGBMRanker:
    """Train a LambdaMART ranker with a proper group-based train/test split."""
    unique_given = df["given_word"].unique()
    rng = np.random.default_rng(0)
    rng.shuffle(unique_given)

    split_idx   = int(len(unique_given) * 0.8)
    train_words = set(unique_given[:split_idx])

    df_rank    = df.sort_values("given_word").copy()
    mask_train = df_rank["given_word"].isin(train_words)
    df_r_train = df_rank[mask_train]
    df_r_test  = df_rank[~mask_train]

    Xr_train, yr_train, g_train = _build_rank_arrays(df_r_train)
    Xr_test,  yr_test,  g_test  = _build_rank_arrays(df_r_test)

    print(f"Ranker train rows: {len(df_r_train)}  |  test rows: {len(df_r_test)}")

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[3, 5, 10],
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    ranker.fit(
        Xr_train, yr_train,
        group=g_train,
        eval_set=[(Xr_test, yr_test)],
        eval_group=[g_test],
        categorical_feature=CAT_COLS,
        callbacks=[
            lgb.early_stopping(stopping_rounds=80, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )
    print(f"Best iteration: {ranker.best_iteration_}")
    return ranker, set(unique_given[split_idx:])  # also return test_words


def _build_rank_arrays(subset: pd.DataFrame):
    Xr = subset[FEATURE_COLS].copy()
    for c in CAT_COLS:
        Xr[c] = Xr[c].astype("category")
    y      = subset["ctr"].values
    groups = subset.groupby("given_word", sort=True).size().tolist()
    return Xr, y, groups


def feature_importance(model, title: str) -> pd.DataFrame:
    """Print and return gain-based feature importance."""
    import pandas as pd
    fi = pd.DataFrame({
        "feature":    model.feature_name_,
        "importance": model.booster_.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    print(f"\n=== {title} ===")
    print(fi.to_string(index=False))
    return fi
