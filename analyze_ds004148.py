import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed"
PROCESSED.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str((PROCESSED / ".matplotlib").resolve()))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

BASE_FEATURE_COLS = [
    "theta_power",
    "alpha_power",
    "beta_power",
    "theta_rel_power",
    "alpha_rel_power",
    "beta_rel_power",
    "alpha_beta_ratio",
    "beta_alpha_ratio",
    "theta_beta_ratio",
    "frontal_theta_power",
    "frontal_theta_rel_power",
    "posterior_alpha_power",
    "posterior_alpha_rel_power",
    "central_beta_power",
    "central_beta_rel_power",
    "posterior_alpha_to_central_beta_ratio",
    "frontal_theta_to_central_beta_ratio",
    "engagement_index",
    "log_theta_power",
    "log_alpha_power",
    "log_beta_power",
]

TEMPORAL_SEED_COLS = [
    "theta_power",
    "alpha_power",
    "beta_power",
    "theta_beta_ratio",
    "alpha_beta_ratio",
    "frontal_theta_power",
    "posterior_alpha_power",
    "central_beta_power",
    "engagement_index",
]

TASK_TO_LABEL = {
    "eyesclosed": 0,
    "mathematic": 1,
}

TASK_TO_DISPLAY = {
    "eyesclosed": "low_engagement_proxy",
    "mathematic": "high_engagement_proxy",
}

REGIONS = {
    "frontal": [
        "Fp1", "Fpz", "Fp2",
        "AF3", "AF4", "AF7", "AF8",
        "Fz", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8",
        "FC1", "FC2", "FC3", "FC4", "FC5", "FC6",
        "FT7", "FT8",
    ],
    "central": [
        "Cz", "C1", "C2", "C3", "C4", "C5", "C6",
        "CPz", "CP1", "CP2", "CP3", "CP4", "CP5", "CP6",
    ],
    "posterior": [
        "Pz", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8",
        "POz", "PO3", "PO4", "PO7", "PO8",
        "Oz", "O1", "O2",
    ],
}


def bandpower(data, sfreq, band, total_band=(1, 40)):
    data = np.asarray(data)
    if data.ndim == 1:
        data = data[:, None]
    if data.shape[0] < data.shape[1]:
        data = data.T

    nperseg = min(int(2 * sfreq), data.shape[0])
    if nperseg < 8:
        raise ValueError("Segment too short for Welch PSD")

    freqs, psd = welch(data, fs=sfreq, nperseg=nperseg, axis=0)
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    total_mask = (freqs >= total_band[0]) & (freqs <= total_band[1])

    bp = np.trapezoid(psd[band_mask], freqs[band_mask], axis=0)
    total = np.trapezoid(psd[total_mask], freqs[total_mask], axis=0)
    return float(np.mean(bp)), float(np.mean(bp / (total + 1e-12)))


def per_channel_bandpower(data, sfreq, band):
    data = np.asarray(data)
    if data.ndim == 1:
        data = data[:, None]
    if data.shape[0] < data.shape[1]:
        data = data.T

    nperseg = min(int(2 * sfreq), data.shape[0])
    if nperseg < 8:
        raise ValueError("Segment too short for Welch PSD")

    freqs, psd = welch(data, fs=sfreq, nperseg=nperseg, axis=0)
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    return np.trapezoid(psd[band_mask], freqs[band_mask], axis=0)


def get_region_subset(data, ch_names, region_name):
    region_channels = REGIONS[region_name]
    idx = [i for i, ch in enumerate(ch_names) if ch in region_channels]
    if not idx:
        raise ValueError(f"No channels found for region {region_name}")
    return data[:, idx]


def compute_features_from_array(data, sfreq, ch_names):
    theta_abs, theta_rel = bandpower(data, sfreq, (4, 7))
    alpha_abs, alpha_rel = bandpower(data, sfreq, (8, 12))
    beta_abs, beta_rel = bandpower(data, sfreq, (13, 30))

    frontal = get_region_subset(data, ch_names, "frontal")
    central = get_region_subset(data, ch_names, "central")
    posterior = get_region_subset(data, ch_names, "posterior")

    frontal_theta_abs, frontal_theta_rel = bandpower(frontal, sfreq, (4, 7))
    posterior_alpha_abs, posterior_alpha_rel = bandpower(posterior, sfreq, (8, 12))
    central_beta_abs, central_beta_rel = bandpower(central, sfreq, (13, 30))

    return {
        "theta_power": theta_abs,
        "alpha_power": alpha_abs,
        "beta_power": beta_abs,
        "theta_rel_power": theta_rel,
        "alpha_rel_power": alpha_rel,
        "beta_rel_power": beta_rel,
        "alpha_beta_ratio": alpha_abs / (beta_abs + 1e-12),
        "beta_alpha_ratio": beta_abs / (alpha_abs + 1e-12),
        "theta_beta_ratio": theta_abs / (beta_abs + 1e-12),
        "frontal_theta_power": frontal_theta_abs,
        "frontal_theta_rel_power": frontal_theta_rel,
        "posterior_alpha_power": posterior_alpha_abs,
        "posterior_alpha_rel_power": posterior_alpha_rel,
        "central_beta_power": central_beta_abs,
        "central_beta_rel_power": central_beta_rel,
        "posterior_alpha_to_central_beta_ratio": posterior_alpha_abs / (central_beta_abs + 1e-12),
        "frontal_theta_to_central_beta_ratio": frontal_theta_abs / (central_beta_abs + 1e-12),
        "engagement_index": beta_abs / (alpha_abs + theta_abs + 1e-12),
        "log_theta_power": np.log1p(theta_abs),
        "log_alpha_power": np.log1p(alpha_abs),
        "log_beta_power": np.log1p(beta_abs),
    }


def available_vhdrs(dataset_root, session, tasks):
    vhdrs = []
    for task in tasks:
        pattern = f"sub-*/ses-{session}/eeg/*task-{task}_eeg.vhdr"
        for path in sorted(dataset_root.glob(pattern)):
            eeg_path = path.with_suffix(".eeg")
            vmrk_path = path.with_suffix(".vmrk")
            if path.exists() and eeg_path.exists() and vmrk_path.exists():
                vhdrs.append(path)
    return vhdrs


def load_windows_from_recording(vhdr_path, task, session, window_seconds, max_windows_per_recording):
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)
    raw.pick("eeg")
    raw.notch_filter(freqs=[60.0], verbose=False)
    raw.filter(l_freq=1.0, h_freq=40.0, verbose=False)
    raw.set_eeg_reference("average", verbose=False)

    data = raw.get_data().T
    ch_names = raw.ch_names
    sfreq = float(raw.info["sfreq"])
    window_size = int(window_seconds * sfreq)
    n_complete_windows = data.shape[0] // window_size
    if max_windows_per_recording is not None:
        n_complete_windows = min(n_complete_windows, max_windows_per_recording)

    subject = vhdr_path.name.split("_")[0]
    rows = []
    for window_idx in range(n_complete_windows):
        start = window_idx * window_size
        stop = start + window_size
        segment = data[start:stop]
        feats = compute_features_from_array(segment, sfreq, ch_names)
        feats.update(
            {
                "dataset": "ds004148",
                "subject": subject,
                "session": session,
                "task": task,
                "condition": task,
                "condition_display": TASK_TO_DISPLAY[task],
                "segment_id": f"{subject}_{session}_{task}_win{window_idx:03d}",
                "window_idx": window_idx,
                "time_start_seconds": start / sfreq,
                "sfreq": sfreq,
                "window_seconds": window_seconds,
                "source_file": str(vhdr_path),
                "label": TASK_TO_LABEL[task],
            }
        )
        rows.append(feats)
    return rows


def build_feature_table(dataset_root, session, tasks, window_seconds, max_windows_per_recording):
    rows = []
    vhdrs = available_vhdrs(dataset_root, session=session, tasks=tasks)
    if not vhdrs:
        raise FileNotFoundError(
            "No downloaded BrainVision task files were found. "
            "Make sure git-annex content has been fetched for the requested tasks."
        )

    for vhdr_path in vhdrs:
        task = vhdr_path.name.split("_task-")[1].split("_eeg.vhdr")[0]
        rows.extend(
            load_windows_from_recording(
                vhdr_path=vhdr_path,
                task=task,
                session=session,
                window_seconds=window_seconds,
                max_windows_per_recording=max_windows_per_recording,
            )
        )
    return pd.DataFrame(rows)


def save_boxplots(df, output_dir):
    for metric in ["alpha_power", "beta_power", "alpha_beta_ratio"]:
        groups = [df.loc[df["task"] == task, metric].dropna().values for task in sorted(df["task"].unique())]
        labels = list(sorted(df["task"].unique()))
        plt.figure(figsize=(9, 5))
        plt.boxplot(groups, tick_labels=labels)
        plt.title(f"{metric} by task")
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"ds004148_{metric}_boxplot.png", dpi=150)
        plt.close()


def save_scatter(df, output_dir):
    plt.figure(figsize=(8, 6))
    for task in sorted(df["task"].unique()):
        sub = df[df["task"] == task]
        plt.scatter(sub["alpha_power"], sub["beta_power"], label=task, alpha=0.7)
    plt.xlabel("Alpha power")
    plt.ylabel("Beta power")
    plt.title("Alpha vs Beta by task")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "ds004148_alpha_vs_beta_scatter.png", dpi=150)
    plt.close()


def save_xgboost_feature_importance(model, feature_cols, output_dir):
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
    plt.figure(figsize=(10, 7))
    plt.barh(importances.index, importances.values)
    plt.xlabel("XGBoost feature importance")
    plt.title("Feature importance for proxy engagement classification")
    plt.tight_layout()
    plt.savefig(output_dir / "ds004148_xgboost_feature_importance.png", dpi=150)
    plt.close()
    importances.sort_values(ascending=False).to_csv(
        output_dir / "ds004148_xgboost_feature_importance.csv",
        header=["importance"],
        index_label="feature",
    )


def exponential_moving_average(values, alpha=0.35):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values

    smoothed = np.empty_like(values)
    smoothed[0] = values[0]
    for idx in range(1, len(values)):
        smoothed[idx] = alpha * values[idx] + (1 - alpha) * smoothed[idx - 1]
    return smoothed


def save_engagement_score_plot(model, df, test_subjects, feature_cols, output_dir, smoothing_alpha=0.35):
    subject = sorted(test_subjects)[0]
    subject_df = df[df["subject"] == subject].copy()
    if subject_df.empty:
        return None, None

    score_rows = []

    plt.figure(figsize=(10, 5))
    for task in sorted(subject_df["task"].unique()):
        task_df = subject_df[subject_df["task"] == task].sort_values("time_start_seconds").copy()
        raw_scores = model.predict_proba(task_df[feature_cols])[:, 1]
        smoothed_scores = exponential_moving_average(raw_scores, alpha=smoothing_alpha)

        task_score_df = task_df[
            ["subject", "session", "task", "segment_id", "window_idx", "time_start_seconds"]
        ].copy()
        task_score_df["xgboost_raw_high_engagement_probability"] = raw_scores
        task_score_df["xgboost_smoothed_high_engagement_probability"] = smoothed_scores
        task_score_df["smoothing_alpha"] = smoothing_alpha
        score_rows.append(task_score_df)

        plt.plot(
            task_df["time_start_seconds"] / 60,
            raw_scores,
            marker="o",
            linestyle=":",
            linewidth=1,
            alpha=0.35,
            label=f"{task} raw",
        )
        plt.plot(
            task_df["time_start_seconds"] / 60,
            smoothed_scores,
            marker="o",
            linewidth=2,
            label=f"{task} smoothed",
        )

    plt.ylim(-0.05, 1.05)
    plt.xlabel("Time in recording (minutes)")
    plt.ylabel("XGBoost P(high-engagement proxy)")
    plt.title(f"Smoothed downstream engagement proxy score over time ({subject})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "ds004148_xgboost_engagement_score_over_time.png", dpi=150)
    plt.close()

    scores_df = pd.concat(score_rows, ignore_index=True)
    scores_df.to_csv(output_dir / "ds004148_xgboost_engagement_score_over_time.csv", index=False)
    return subject, smoothing_alpha


def compute_task_channel_bandpower(dataset_root, session, tasks, window_seconds, max_windows_per_recording):
    task_values = {
        task: {
            "alpha": [],
            "beta": [],
            "beta_alpha_ratio": [],
        }
        for task in tasks
    }
    ch_names = None
    sfreq_seen = None

    for vhdr_path in available_vhdrs(dataset_root, session=session, tasks=tasks):
        task = vhdr_path.name.split("_task-")[1].split("_eeg.vhdr")[0]
        raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)
        raw.pick("eeg")
        raw.notch_filter(freqs=[60.0], verbose=False)
        raw.filter(l_freq=1.0, h_freq=40.0, verbose=False)
        raw.set_eeg_reference("average", verbose=False)

        data = raw.get_data().T
        if ch_names is None:
            ch_names = raw.ch_names
            sfreq_seen = float(raw.info["sfreq"])

        sfreq = float(raw.info["sfreq"])
        window_size = int(window_seconds * sfreq)
        n_complete_windows = data.shape[0] // window_size
        if max_windows_per_recording is not None:
            n_complete_windows = min(n_complete_windows, max_windows_per_recording)

        for window_idx in range(n_complete_windows):
            start = window_idx * window_size
            stop = start + window_size
            segment = data[start:stop]
            alpha = per_channel_bandpower(segment, sfreq, (8, 12))
            beta = per_channel_bandpower(segment, sfreq, (13, 30))
            task_values[task]["alpha"].append(alpha)
            task_values[task]["beta"].append(beta)
            task_values[task]["beta_alpha_ratio"].append(beta / (alpha + 1e-12))

    means = {}
    for task, band_values in task_values.items():
        means[task] = {}
        for band, values in band_values.items():
            if values:
                means[task][band] = np.mean(values, axis=0)

    return means, ch_names, sfreq_seen


def save_topomap(values, ch_names, sfreq, title, output_path):
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    info.set_montage("standard_1020", on_missing="ignore")
    montage = info.get_montage()
    pos_names = set(montage.ch_names) if montage is not None else set()
    keep_idx = [i for i, ch in enumerate(ch_names) if ch in pos_names]
    keep_names = [ch_names[i] for i in keep_idx]
    keep_values = np.asarray(values)[keep_idx]

    plot_info = mne.create_info(ch_names=keep_names, sfreq=sfreq, ch_types="eeg")
    plot_info.set_montage("standard_1020", on_missing="ignore")

    fig, ax = plt.subplots(figsize=(6, 5))
    im, _ = mne.viz.plot_topomap(keep_values, plot_info, axes=ax, show=False)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_task_difference_topomaps(dataset_root, session, tasks, window_seconds, max_windows_per_recording, output_dir):
    required = {"eyesclosed", "mathematic"}
    if not required.issubset(set(tasks)):
        return

    means, ch_names, sfreq = compute_task_channel_bandpower(
        dataset_root=dataset_root,
        session=session,
        tasks=["eyesclosed", "mathematic"],
        window_seconds=window_seconds,
        max_windows_per_recording=max_windows_per_recording,
    )
    if ch_names is None or not means["eyesclosed"] or not means["mathematic"]:
        return

    alpha_diff = means["eyesclosed"]["alpha"] - means["mathematic"]["alpha"]
    beta_diff = means["mathematic"]["beta"] - means["eyesclosed"]["beta"]
    ratio_diff = means["mathematic"]["beta_alpha_ratio"] - means["eyesclosed"]["beta_alpha_ratio"]

    save_topomap(
        alpha_diff,
        ch_names,
        sfreq,
        "Alpha power difference: eyesclosed minus mathematic",
        output_dir / "ds004148_topomap_alpha_eyesclosed_minus_mathematic.png",
    )
    save_topomap(
        beta_diff,
        ch_names,
        sfreq,
        "Beta power difference: mathematic minus eyesclosed",
        output_dir / "ds004148_topomap_beta_mathematic_minus_eyesclosed.png",
    )
    save_topomap(
        ratio_diff,
        ch_names,
        sfreq,
        "Beta/alpha ratio difference: mathematic minus eyesclosed",
        output_dir / "ds004148_topomap_beta_alpha_ratio_mathematic_minus_eyesclosed.png",
    )


def split_by_subject(df, test_size=0.25, random_state=42):
    subject_task = (
        df.groupby(["subject", "label"]).size().unstack(fill_value=0).reset_index()
    )
    eligible_subjects = subject_task[
        (subject_task[0] > 0) & (subject_task[1] > 0)
    ]["subject"].tolist()

    if len(eligible_subjects) < 8:
        raise ValueError("Need at least 8 subjects with both classes present for a subject split.")

    train_subjects, test_subjects = train_test_split(
        eligible_subjects,
        test_size=test_size,
        random_state=random_state,
    )

    train_df = df[df["subject"].isin(train_subjects)].copy()
    test_df = df[df["subject"].isin(test_subjects)].copy()
    return train_df, test_df, train_subjects, test_subjects


def fit_and_evaluate_model(model_name, clf, X_train, y_train, X_test, y_test, output_dir):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["low_engagement_proxy", "high_engagement_proxy"],
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["low_engagement_proxy", "high_engagement_proxy"],
    )
    disp.plot()
    plt.title(f"ds004148 confusion matrix: {model_name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"ds004148_{model_name}_confusion_matrix.png", dpi=150)
    plt.close()
    return report, clf


def add_temporal_features(df, feature_cols, n_lags=2):
    df = df.sort_values(["subject", "task", "window_idx"]).copy()
    temporal_cols = []
    for col in feature_cols:
        for lag in range(1, n_lags + 1):
            lag_col = f"{col}_lag{lag}"
            df[lag_col] = df.groupby(["subject", "task"])[col].shift(lag)
            temporal_cols.append(lag_col)

        delta_col = f"{col}_delta1"
        df[delta_col] = df[col] - df.groupby(["subject", "task"])[col].shift(1)
        temporal_cols.append(delta_col)

    return df, temporal_cols


def tune_xgboost_with_group_cv(X, y, groups):
    base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
    )
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions={
            "n_estimators": [150, 250, 350, 450],
            "max_depth": [2, 3, 4, 5],
            "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.12],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "min_child_weight": [1, 3, 5, 7],
            "gamma": [0.0, 0.1, 0.3],
        },
        n_iter=24,
        scoring="f1_macro",
        cv=GroupKFold(n_splits=5),
        random_state=42,
        refit=True,
    )
    search.fit(X, y, groups=groups)
    return search.best_estimator_, search.best_score_, search.best_params_


def run_classifiers(df, output_dir):
    legacy_cm_path = output_dir / "ds004148_confusion_matrix.png"
    if legacy_cm_path.exists():
        legacy_cm_path.unlink()

    cm_paths = [
        output_dir / "ds004148_logistic_regression_confusion_matrix.png",
        output_dir / "ds004148_random_forest_confusion_matrix.png",
        output_dir / "ds004148_xgboost_confusion_matrix.png",
        legacy_cm_path,
    ]
    class_counts = df["label"].value_counts()
    if len(class_counts) < 2 or class_counts.min() < 8:
        for path in cm_paths:
            if path.exists():
                path.unlink()
        return {
            "skipped": True,
            "reason": "Need at least 8 windows in each class before fitting the classifier.",
            "class_counts": class_counts.to_dict(),
        }

    try:
        train_df, test_df, train_subjects, test_subjects = split_by_subject(df)
    except ValueError as exc:
        for path in cm_paths:
            if path.exists():
                path.unlink()
        return {
            "skipped": True,
            "reason": str(exc),
        }

    train_class_counts = train_df["label"].value_counts()
    test_class_counts = test_df["label"].value_counts()
    if len(train_class_counts) < 2 or len(test_class_counts) < 2:
        for path in cm_paths:
            if path.exists():
                path.unlink()
        return {
            "skipped": True,
            "reason": "Subject split produced a train or test set without both classes.",
            "train_class_counts": train_class_counts.to_dict(),
            "test_class_counts": test_class_counts.to_dict(),
        }

    full_df, temporal_cols = add_temporal_features(df, TEMPORAL_SEED_COLS, n_lags=2)
    feature_cols = BASE_FEATURE_COLS + temporal_cols

    train_df = full_df[full_df["subject"].isin(train_subjects)].copy()
    test_df = full_df[full_df["subject"].isin(test_subjects)].copy()
    train_df = train_df.dropna(subset=feature_cols)
    test_df = test_df.dropna(subset=feature_cols)

    X_train = train_df[feature_cols]
    y_train = train_df["label"]
    X_test = test_df[feature_cols]
    y_test = test_df["label"]

    logistic_clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )

    tuned_xgb, best_cv_f1_macro, best_params = tune_xgboost_with_group_cv(
        X_train, y_train, train_df["subject"]
    )

    logistic_report, logistic_model = fit_and_evaluate_model(
        "logistic_regression",
        logistic_clf,
        X_train,
        y_train,
        X_test,
        y_test,
        output_dir,
    )

    random_forest_clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1,
    )
    random_forest_report, random_forest_model = fit_and_evaluate_model(
        "random_forest",
        random_forest_clf,
        X_train,
        y_train,
        X_test,
        y_test,
        output_dir,
    )

    xgboost_report, xgboost_model = fit_and_evaluate_model(
        "xgboost",
        tuned_xgb,
        X_train,
        y_train,
        X_test,
        y_test,
        output_dir,
    )
    save_xgboost_feature_importance(xgboost_model, feature_cols, output_dir)
    score_subject, smoothing_alpha = save_engagement_score_plot(
        xgboost_model, full_df, test_subjects, feature_cols, output_dir
    )

    return {
        "split_mode": "subject",
        "n_train_subjects": len(train_subjects),
        "n_test_subjects": len(test_subjects),
        "train_subjects": sorted(train_subjects),
        "test_subjects": sorted(test_subjects),
        "train_class_counts": train_class_counts.to_dict(),
        "test_class_counts": test_class_counts.to_dict(),
        "engagement_score_subject": score_subject,
        "engagement_score_smoothing": "exponential_moving_average",
        "engagement_score_smoothing_alpha": smoothing_alpha,
        "feature_columns_used": feature_cols,
        "xgboost_group_cv_best_f1_macro": float(best_cv_f1_macro),
        "xgboost_best_params": best_params,
        "models": {
            "logistic_regression": logistic_report,
            "random_forest": random_forest_report,
            "xgboost": xgboost_report,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="data/ds004148")
    parser.add_argument("--session", default="session1")
    parser.add_argument("--window-seconds", type=int, default=10)
    parser.add_argument("--max-windows-per-recording", type=int, default=None)
    parser.add_argument("--tasks", nargs="+", default=["eyesclosed", "mathematic"])
    parser.add_argument(
        "--reuse-feature-table",
        action="store_true",
        help="Reuse processed/ds004148/ds004148_features.csv instead of rereading raw EEG files.",
    )
    args = parser.parse_args()

    dataset_root = (ROOT / args.dataset_root).resolve()
    output_dir = PROCESSED / "ds004148"
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_table_path = output_dir / "ds004148_features.csv"

    if args.reuse_feature_table:
        if not feature_table_path.exists():
            raise FileNotFoundError(f"Cannot reuse missing feature table: {feature_table_path}")
        df = pd.read_csv(feature_table_path)
    else:
        df = build_feature_table(
            dataset_root=dataset_root,
            session=args.session,
            tasks=args.tasks,
            window_seconds=args.window_seconds,
            max_windows_per_recording=args.max_windows_per_recording,
        )
        df.to_csv(feature_table_path, index=False)

    save_boxplots(df, output_dir)
    save_scatter(df, output_dir)
    if not args.reuse_feature_table:
        save_task_difference_topomaps(
            dataset_root=dataset_root,
            session=args.session,
            tasks=args.tasks,
            window_seconds=args.window_seconds,
            max_windows_per_recording=args.max_windows_per_recording,
            output_dir=output_dir,
        )
    report = run_classifiers(df, output_dir)

    summary = {
        "n_rows": int(len(df)),
        "n_subjects": int(df["subject"].nunique()),
        "task_counts": df["task"].value_counts().to_dict(),
        "session": args.session,
        "window_seconds": args.window_seconds,
        "classification_report": report,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("Saved feature table to", output_dir / "ds004148_features.csv")
    print("Task counts:")
    print(df["task"].value_counts())
    print("Saved plots and summary to", output_dir)


if __name__ == "__main__":
    main()
