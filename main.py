import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist, squareform
from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score, ConfusionMatrixDisplay
from weighted_levenshtein import lev
import ruptures as rpt
from fastcluster import linkage
import matplotlib.pyplot as plt
import yaml
from joblib import Parallel, delayed
import argparse


# --------------------------------------------------------------------
# Utilities for hierarchical clustering / ordering
# --------------------------------------------------------------------


def seriation(Z, n_leaves, cur_index):
    """Recover the leaf ordering from a hierarchical clustering tree."""
    if cur_index < n_leaves:
        return [cur_index]

    left = int(Z[cur_index - n_leaves, 0])
    right = int(Z[cur_index - n_leaves, 1])
    return seriation(Z, n_leaves, left) + seriation(Z, n_leaves, right)


def compute_serial_matrix(dist_mat, method="ward"):
    """Reorder a distance matrix by hierarchical clustering."""
    n = len(dist_mat)
    flat_dist = squareform(dist_mat)
    Z = linkage(flat_dist, method=method, preserve_input=True)
    order = seriation(Z, n, n + n - 2)

    seriated = np.zeros((n, n))
    a, b = np.triu_indices(n, k=1)
    seriated[a, b] = dist_mat[[order[i] for i in a], [order[j] for j in b]]
    seriated[b, a] = seriated[a, b]

    return seriated, order, Z


# --------------------------------------------------------------------
# Edit distance between symbolic sequences
# --------------------------------------------------------------------


def generalized_edit_distance(seq1, seq2, d_m, deletion_cost, insertion_cost):
    """Weighted Levenshtein between two sequences, normalized by max length."""
    n1, n2 = len(seq1), len(seq2)
    s1, s2 = "".join(seq1), "".join(seq2)

    sub_costs = np.ones((128, 128), dtype=np.float64)
    k = d_m.shape[0]
    sub_costs[65:65 + k, 65:65 + k] = d_m

    ins_costs = np.ones(128, dtype=np.float64) * insertion_cost
    del_costs = np.ones(128, dtype=np.float64) * deletion_cost

    dist = lev(
        s1,
        s2,
        insert_costs=ins_costs,
        delete_costs=del_costs,
        substitute_costs=sub_costs,
    )
    return dist / max(n1, n2, 1)


def _normalize_distance_train(D):
    """Median-based scaling on TRAIN×TRAIN block."""
    tri = D[np.triu_indices_from(D, 1)]
    m = np.median(tri)
    scale = m if m > 0 else 1.0
    return D / scale, scale


def _apply_scale_to_test_block(D_te_tr, scale):
    """Apply train-derived scale to TEST×TRAIN block."""
    return D_te_tr / (scale if scale > 0 else 1.0)


def _softmax(X, axis=-1, temp=1.0):
    """Temperature-scaled softmax."""
    temp = max(temp, 1e-8)
    Z = X / temp
    Z -= np.max(Z, axis=axis, keepdims=True)
    E = np.exp(Z)
    return E / np.sum(E, axis=axis, keepdims=True)


# --------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------


def load_raw_features(config, path):
    """
    Load per-modality CSVs and keep only (subject, label) pairs where
    all modalities have enough available samples.
    """
    thr = config["general"]["available_segment_prop"]
    kept_records = []

    subjects = [
        "1030", "1105", "1106", "1241", "1271", "1314", "1323",
        "1337", "1372", "1417", "1434", "1544", "1547", "1595",
        "1629", "1716", "1717", "1744", "1868", "1892", "1953",
    ]
    label_set = [str(i) for i in range(1, 10)]

    modalities = ["oculomotor", "scanpath", "AoI", "eda", "ecg"]

    # First pass: select (subject, label) pairs that pass the availability threshold
    for subject in subjects:
        for label in label_set:
            ok = True
            for mod in modalities:
                fname = f"{subject}_{label}_{mod}.csv"
                try:
                    df_mod = pd.read_csv(path + fname)
                    df_mod = df_mod.interpolate(axis=0).ffill().bfill()
                    col = df_mod.iloc[:, 1].to_numpy()
                    prop_avail = np.count_nonzero(~np.isnan(col)) / len(df_mod)
                    if prop_avail < thr:
                        ok = False
                        break
                except Exception:
                    ok = False
                    break
            if ok:
                kept_records.append(f"{subject}_{label}")

    # Second pass: actually load the data for valid (subject, label, modality)
    raw_data = {}
    for rec in kept_records:
        subject, label = rec.split("_")
        for modality in modalities:
            fname = f"{subject}_{label}_{modality}.csv"
            try:
                df_mod = pd.read_csv(path + fname)
                df_mod = df_mod.interpolate(axis=0).ffill().bfill()
                raw_data[(subject, label, modality)] = df_mod
            except Exception:
                # If one file is missing here, we simply skip that modality for this recording
                pass

    return raw_data


# --------------------------------------------------------------------
# ECDF normalization
# --------------------------------------------------------------------


class ECDFNormalizer:
    """Learn per-feature ECDF on train set and apply to train/test."""

    def __init__(self, config):
        self.config = config
        self.norm_map = {}

    def _feature_list_for_modality(self, modality):
        data_cfg = self.config["data"]
        if modality == "oculomotor":
            return data_cfg["oculomotor_features"]
        if modality == "scanpath":
            return data_cfg["scanpath_features"]
        if modality == "AoI":
            return data_cfg["aoi_features"]
        if modality == "eda":
            return data_cfg["eda_features"]
        if modality == "ecg":
            return data_cfg["ecg_features"]
        raise ValueError(f"Unknown modality {modality}")

    def fit(self, raw_data, train_records):
        self.norm_map.clear()

        modalities = ["oculomotor", "scanpath", "AoI", "eda", "ecg"]

        for modality in modalities:
            feats = self._feature_list_for_modality(modality)
            for feat in feats:
                if feat == "startTime(s)":
                    continue

                vals_all = []
                for subj, lab in train_records:
                    key = (subj, lab, modality)
                    if key not in raw_data:
                        continue
                    vals_all.extend(raw_data[key][feat].to_numpy())

                if not vals_all:
                    continue

                ecdf_obj = stats.ecdf(vals_all).cdf
                self.norm_map[(modality, feat)] = ecdf_obj

    def transform(self, raw_data, records_subset):
        norm_data = {}
        modalities = ["oculomotor", "scanpath", "AoI", "eda", "ecg"]

        for subj, lab in records_subset:
            for modality in modalities:
                key = (subj, lab, modality)
                if key not in raw_data:
                    continue

                df_in = raw_data[key]
                feats = self._feature_list_for_modality(modality)
                new_cols = {}

                for feat in feats:
                    if feat == "startTime(s)":
                        new_cols[feat] = df_in[feat].to_numpy()
                        continue

                    x = df_in[feat].to_numpy()
                    ecdf_fun = self.norm_map.get((modality, feat))
                    new_cols[feat] = x if ecdf_fun is None else ecdf_fun.evaluate(x)

                norm_data[(subj, lab, modality)] = pd.DataFrame(new_cols)

        return norm_data


# --------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------


class Segmenter:
    """Ruptures/PELT segmentation on each normalized recording."""

    def __init__(self):
        pass

    def _segment_signal(self, X):
        pen = np.log(len(X)) / 10.0
        algo = rpt.Pelt(model="l2", jump=1).fit(X)
        bkps = algo.predict(pen=pen)
        bkps.insert(0, 0)
        return bkps

    def segment_records(self, norm_data):
        segments = {}

        for (subj, lab, modality), df_mod in norm_data.items():
            if modality == "oculomotor":
                fix_cols = [c for c in df_mod.columns if c.startswith("fix")]
                if fix_cols:
                    X_fix = df_mod[fix_cols].to_numpy()
                    segments[(subj, lab, "oculomotorFixation")] = self._segment_signal(X_fix)

                sac_cols = [c for c in df_mod.columns if c.startswith("sac")]
                if sac_cols:
                    X_sac = df_mod[sac_cols].to_numpy()
                    segments[(subj, lab, "oculomotorSaccade")] = self._segment_signal(X_sac)
            else:
                feat_cols = [c for c in df_mod.columns if c != "startTime(s)"]
                if not feat_cols:
                    continue
                X_all = df_mod[feat_cols].to_numpy()
                segments[(subj, lab, modality)] = self._segment_signal(X_all)

        return segments


# --------------------------------------------------------------------
# Symbolization (segment clustering)
# --------------------------------------------------------------------


class Symbolizer:
    """Convert segments to discrete symbols using train-only models."""

    def __init__(self, config):
        self.config = config
        self.models = {}

    def _n_centers(self, submodality):
        nb = self.config["symbolization"]["nb_clusters"]
        if submodality == "scanpath":
            return nb["scanpath"]
        if submodality == "AoI":
            return nb["aoi"]
        if submodality == "eda":
            return nb["eda"]
        if submodality == "ecg":
            return nb["ecg"]
        return nb["oculomotor"]

    @staticmethod
    def _features_for_submodality(submodality, df_mod):
        if submodality == "oculomotorFixation":
            return [c for c in df_mod.columns if c.startswith("fix")]
        if submodality == "oculomotorSaccade":
            return [c for c in df_mod.columns if c.startswith("sac")]
        if submodality == "scanpath":
            return [c for c in df_mod.columns if c.startswith("Sp")]
        if submodality == "AoI":
            return [c for c in df_mod.columns if c.startswith("AoI")]
        if submodality == "eda":
            return [c for c in df_mod.columns if c.startswith("eda")]
        if submodality == "ecg":
            return [c for c in df_mod.columns if c.startswith("ecg")]
        return [c for c in df_mod.columns if c != "startTime(s)"]

    def _collect_segment_means(self, norm_data, segments, records_subset, submodality):
        all_means = []

        for subj, lab in records_subset:
            seg_key = (subj, lab, submodality)
            if seg_key not in segments:
                continue

            bkps = segments[seg_key]
            modality = "oculomotor" if submodality.startswith("oculomotor") else submodality
            data_key = (subj, lab, modality)
            if data_key not in norm_data:
                continue

            df_mod = norm_data[data_key]
            cols = self._features_for_submodality(submodality, df_mod)
            if not cols:
                continue

            X = df_mod[cols].to_numpy()
            for i in range(1, len(bkps)):
                start, end = bkps[i - 1], bkps[i]
                seg_mean = np.mean(X[start:end, :], axis=0)
                all_means.append(seg_mean)

        return np.array(all_means)

    def fit(self, norm_data_train, segments_train, train_records):
        self.models.clear()
        submodalities = [
            "oculomotorFixation",
            "oculomotorSaccade",
            "scanpath",
            "AoI",
            "eda",
            "ecg",
        ]

        for sm in submodalities:
            k = self._n_centers(sm)
            all_means = self._collect_segment_means(
                norm_data_train, segments_train, train_records, sm
            )
            if all_means.size == 0:
                continue

            kpca = KernelPCA(n_components=10, kernel="rbf", n_jobs=-1)
            Z = kpca.fit_transform(all_means)

            kmeans = KMeans(n_clusters=k, n_init=100, random_state=0)
            kmeans.fit(Z)

            centers = kmeans.cluster_centers_
            dist_mat = cdist(centers, centers)
            _, order, _ = compute_serial_matrix(dist_mat, "ward")

            inv_order = np.zeros(len(order), dtype=int)
            for new_idx, old_idx in enumerate(order):
                inv_order[old_idx] = new_idx

            centers_reordered = centers[order]

            self.models[sm] = {
                "kpca": kpca,
                "kmeans": kmeans,
                "inv_reorder": inv_order,
                "centers": centers_reordered,
            }

    def transform(self, norm_data, segments, records_subset):
        symb_out = {}

        for sm, model in self.models.items():
            kpca = model["kpca"]
            kmeans = model["kmeans"]
            inv_reorder = model["inv_reorder"]
            centers_reordered = model["centers"]

            recordings_dict = {}

            for subj, lab in records_subset:
                seg_key = (subj, lab, sm)
                if seg_key not in segments:
                    continue

                bkps = segments[seg_key]
                modality = "oculomotor" if sm.startswith("oculomotor") else sm
                data_key = (subj, lab, modality)
                if data_key not in norm_data:
                    continue

                df_mod = norm_data[data_key]
                cols = self._features_for_submodality(sm, df_mod)
                if not cols:
                    continue

                X = df_mod[cols].to_numpy()
                seg_means, seg_lens = [], []

                for i in range(1, len(bkps)):
                    start, end = bkps[i - 1], bkps[i]
                    seg_mean = np.mean(X[start:end, :], axis=0)
                    seg_means.append(seg_mean)
                    seg_lens.append(end - start)

                if not seg_means:
                    continue

                Z = kpca.transform(np.array(seg_means))
                raw_labels = kmeans.predict(Z)
                remapped = [int(inv_reorder[l]) for l in raw_labels]

                rec_name = f"{subj}_{lab}"
                recordings_dict[rec_name] = {
                    "sequence": remapped,
                    "lengths": seg_lens,
                }

            symb_out[sm] = {
                "centers": centers_reordered,
                "recordings": recordings_dict,
            }

        return symb_out


# --------------------------------------------------------------------
# Distances per modality (symbol sequences)
# --------------------------------------------------------------------


def build_distance_matrices_per_modality(symb_results, records, config, binning=True):
    """Build weighted-Levenshtein distance matrices per submodality."""
    dist_dict = {}
    n = len(records)

    del_cost = config["clustering"]["edit_distance"]["deletion_cost"]
    ins_cost = config["clustering"]["edit_distance"]["insertion_cost"]

    for sm, pack in symb_results.items():
        centers = pack["centers"]
        recs = pack["recordings"]

        d_m = cdist(centers, centers)
        max_dm = np.max(d_m)
        if max_dm > 0:
            d_m /= max_dm

        seq_map = {}
        for rec in records:
            if rec not in recs:
                seq_map[rec] = []
                continue

            labs = recs[rec]["sequence"]
            lens = recs[rec]["lengths"]

            if binning:
                expanded = []
                for lab_id, seg_len in zip(labs, lens):
                    expanded.extend([chr(lab_id + 65)] * seg_len)
            else:
                expanded = [chr(lab_id + 65) for lab_id in labs]

            seq_map[rec] = expanded

        D = np.zeros((n, n), dtype=float)
        for j in range(1, n):
            for i in range(j):
                s1 = seq_map[records[i]]
                s2 = seq_map[records[j]]
                dij = generalized_edit_distance(
                    s1,
                    s2,
                    d_m,
                    deletion_cost=del_cost,
                    insertion_cost=ins_cost,
                )
                D[i, j] = D[j, i] = dij

        dist_dict[sm] = D

    return dist_dict


# --------------------------------------------------------------------
# Gating features and models
# --------------------------------------------------------------------


def _row_knn_indices_and_stats(D, k=5, is_square=True):
    """Return k-NN indices, mean, std per row."""
    n, r = D.shape
    k_eff = min(k, r - (1 if is_square else 0))

    nn_idx_list = []
    mean_k = np.zeros(n, dtype=float)
    std_k = np.zeros(n, dtype=float)

    for i in range(n):
        row = D[i].copy()
        if is_square:
            row[i] = np.inf

        kk = max(1, k_eff)
        idx = np.argpartition(row, kk - 1)[:kk]
        vals = row[idx]

        nn_idx_list.append(idx)
        mean_k[i] = vals.mean()
        std_k[i] = vals.std()

    return nn_idx_list, mean_k, std_k


def _entropy_of_labels(labels, class_list):
    """Normalized entropy of labels over a fixed class set."""
    if not labels:
        return 1.0

    k = len(class_list)
    counts = {c: 0 for c in class_list}
    for y in labels:
        counts[y] = counts.get(y, 0) + 1

    p = np.array([counts[c] for c in class_list], dtype=float)
    p /= (p.sum() + 1e-12)

    ent = -(p * np.log(p + 1e-12)).sum()
    return float(ent / (np.log(k) + 1e-12))


def _gate_features_block_scale_free(
    Ds_block,
    modalities,
    y_ref,
    k=5,
    is_square=True,
    class_list=None,
):
    """
    Build 3 scale-free features per modality, per sample:
      - f1: inverse ratio (mean kNN distance / row median)
      - f3: inverse coefficient of variation of kNN distances
      - f4: 1 - normalized entropy of kNN labels
    """
    if class_list is None:
        class_list = sorted(list(np.unique(y_ref)))

    feats_all = []
    mod2cols = {}
    col = 0
    eps = 1e-12

    for m in modalities:
        D = Ds_block[m]
        n, _ = D.shape

        nn_list, mean_k, std_k = _row_knn_indices_and_stats(
            D,
            k=k,
            is_square=is_square,
        )

        row_median = np.zeros(n, dtype=float)
        for i in range(n):
            row = D[i].copy()
            if is_square:
                row[i] = np.nan
            row_median[i] = np.nanmedian(row)

        ent = np.zeros(n, dtype=float)
        for i, nn_idx in enumerate(nn_list):
            ent[i] = _entropy_of_labels(list(y_ref[nn_idx]), class_list)

        ratio = mean_k / (row_median + eps)
        cv = std_k / (mean_k + eps)
        f1 = 1.0 / (ratio + eps)
        f3 = 1.0 / (cv + eps)
        f4 = 1.0 - ent

        block = np.column_stack([f1, f3, f4])
        feats_all.append(block)
        mod2cols[m] = [col, col + 1, col + 2]
        col += 3

    Phi_full = np.concatenate(feats_all, axis=1) if feats_all else np.zeros((0, 0))
    return Phi_full, mod2cols


def _gate_multilabel_targets_from_KNN_LOO_train(
    Ds_tr_norm,
    modalities,
    y_train,
    k,
    tau,
    return_props=False,
):
    """Build binary reliability targets per modality using LOO kNN agreement."""
    n_tr = len(y_train)
    m = len(modalities)
    k_eff = max(1, min(k, n_tr - 1))

    C_bin = np.zeros((n_tr, m), dtype=int)
    P = np.zeros((n_tr, m), dtype=float)

    for mi, mod in enumerate(modalities):
        D = Ds_tr_norm[mod]
        for i in range(n_tr):
            row = D[i].copy()
            row[i] = np.inf
            nn_idx = np.argpartition(row, k_eff)[:k_eff]
            prop = np.mean(y_train[nn_idx] == y_train[i])
            P[i, mi] = prop
            C_bin[i, mi] = int(prop >= tau)

    if return_props:
        return C_bin, P
    return C_bin


def _fit_gate_multilabel_scale_free(
    Phi_full,
    C_bin,
    mod2cols,
    C=1.0,
    max_iter=1000,
    class_weight="balanced",
    sample_weights_list=None,
):
    """One logistic regression per modality on its 3 features."""
    models = []
    mods_sorted = sorted(mod2cols.keys(), key=str)

    for mi, mod in enumerate(mods_sorted):
        cols = mod2cols[mod]
        X_m = Phi_full[:, cols]
        y_m = C_bin[:, mi]
        sw = None if sample_weights_list is None else sample_weights_list[mi]

        clf = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=max_iter,
            C=C,
            fit_intercept=True,
            class_weight=class_weight,
        )
        clf.fit(X_m, y_m, sample_weight=sw)
        models.append(clf)

    def logits_fn(Phiq_full):
        Z_list = []
        for mi, mod in enumerate(mods_sorted):
            cols = mod2cols[mod]
            X_m = Phiq_full[:, cols]
            z = models[mi].decision_function(X_m)
            Z_list.append(z)
        return np.column_stack(Z_list)

    return models, logits_fn


def _gate_weights_from_logits(Z_gate, temp=1.0, mix_uniform=0.0):
    """Softmax over modality logits with optional uniform mixing."""
    W = _softmax(Z_gate, axis=1, temp=temp)
    if mix_uniform > 0:
        m = W.shape[1]
        W = (1.0 - mix_uniform) * W + mix_uniform * (1.0 / m)
    return W


# --------------------------------------------------------------------
# Kernelization and fusion
# --------------------------------------------------------------------


def _median_sigma_from_dist(D):
    tri = D[np.triu_indices_from(D, 1)]
    tri = tri[np.isfinite(tri)]
    med = np.median(tri) if tri.size > 0 else 1.0
    if not np.isfinite(med) or med <= 1e-12:
        return 1.0
    return med


def _kernelize_blocks(Ds_tr_norm, Ds_te_tr_norm, use_global_sigma=False):
    """
    RBF kernel per modality.
    Returns dicts of train kernels and test-vs-train kernels.
    """
    Ks_tr, Ks_te_tr = {}, {}

    if use_global_sigma:
        meds = [(_median_sigma_from_dist(D)) for D in Ds_tr_norm.values()]
        sig_global = np.mean(meds) if meds else 1.0

    for m, Dtr in Ds_tr_norm.items():
        Dte = Ds_te_tr_norm[m]
        sigma = sig_global if use_global_sigma else _median_sigma_from_dist(Dtr)

        Ktr = np.exp(-(Dtr ** 2) / (sigma ** 2))
        np.fill_diagonal(Ktr, 1.0)
        Kte = np.exp(-(Dte ** 2) / (sigma ** 2))

        Ks_tr[m] = Ktr
        Ks_te_tr[m] = Kte

    return Ks_tr, Ks_te_tr


def fuse_train_pairwise_kernels_weighted_mean(Ks_tr, modalities, W_tr):
    """
    Weighted average of kernels (train × train):
        Kf(i,j) = sum_m ((w_i^m + w_j^m)/2) * K_m(i,j) / sum_m ((w_i^m + w_j^m)/2)
    """
    n_tr = next(iter(Ks_tr.values())).shape[0]
    num = np.zeros((n_tr, n_tr), dtype=float)
    den = np.zeros((n_tr, n_tr), dtype=float)

    for j, m in enumerate(modalities):
        wm = np.maximum(W_tr[:, j], 0.0)
        wpair = 0.5 * (wm[:, None] + wm[None, :])
        num += wpair * Ks_tr[m]
        den += wpair

    den = np.where(den <= 1e-12, 1.0, den)
    Kf = num / den
    np.fill_diagonal(Kf, 1.0)
    return Kf


def fuse_test_pairwise_kernels_weighted_mean(Ks_te_tr, modalities, W_te, W_tr):
    """
    Weighted average of kernels (test × train):
        Kf(i,j) = sum_m ((w_i^m + w_j^m)/2) * K_m(i,j) / sum_m ((w_i^m + w_j^m)/2)
    """
    n_te, n_tr = next(iter(Ks_te_tr.values())).shape
    num = np.zeros((n_te, n_tr), dtype=float)
    den = np.zeros((n_te, n_tr), dtype=float)

    for j, m in enumerate(modalities):
        wi = np.maximum(W_te[:, j], 0.0)[:, None]
        wj = np.maximum(W_tr[:, j], 0.0)[None, :]
        wpair = 0.5 * (wi + wj)

        num += wpair * Ks_te_tr[m]
        den += wpair

    den = np.where(den <= 1e-12, 1.0, den)
    return num / den


# --------------------------------------------------------------------
# One LOSO fold
# --------------------------------------------------------------------


def run_loso_fold(
    train_index,
    test_index,
    config,
    raw_data,
    all_records,
    y_all,
    conditions,
    conditions_dict,
    task
):
    """Run one LOSO fold and return metrics + confusion + mean test weights."""

    train_records = [all_records[i] for i in train_index]
    test_records = [all_records[i] for i in test_index]
    y_train = y_all[train_index]
    y_test = y_all[test_index]

    gate_temp = 0.5 if task == "ternary" else 1.0

    train_pairs = [tuple(r.split("_")) for r in train_records]
    test_pairs = [tuple(r.split("_")) for r in test_records]

    # 1. ECDF normalization
    ecdf_norm = ECDFNormalizer(config)
    ecdf_norm.fit(raw_data, train_pairs)
    norm_train = ecdf_norm.transform(raw_data, train_pairs)
    norm_test = ecdf_norm.transform(raw_data, test_pairs)

    # 2. Segmentation
    segm = Segmenter()
    segments_train = segm.segment_records(norm_train)
    segments_test = segm.segment_records(norm_test)

    # 3. Symbolization
    symb = Symbolizer(config)
    symb.fit(norm_train, segments_train, train_pairs)
    symb_train = symb.transform(norm_train, segments_train, train_pairs)
    symb_test = symb.transform(norm_test, segments_test, test_pairs)

    # 4. Distances per modality
    union_records = train_records + test_records
    merged_symb = {}

    for sm in symb_train:
        merged_symb[sm] = {
            "centers": symb_train[sm]["centers"],
            "recordings": {},
        }
        merged_symb[sm]["recordings"].update(symb_train[sm]["recordings"])
        if sm in symb_test:
            merged_symb[sm]["recordings"].update(symb_test[sm]["recordings"])

    dist_all = build_distance_matrices_per_modality(
        merged_symb,
        union_records,
        config,
        binning=True,
    )
    modalities = list(dist_all.keys())

    # 5. Slice TRAIN/TEST blocks
    n_tr = len(train_records)
    n_te = len(test_records)

    idx_train = np.arange(n_tr)
    idx_test = np.arange(n_tr, n_tr + n_te)

    Ds_tr = {m: dist_all[m][np.ix_(idx_train, idx_train)] for m in modalities}
    Ds_te_tr = {m: dist_all[m][np.ix_(idx_test, idx_train)] for m in modalities}

    # Train-only scaling per modality
    Ds_tr_norm, scales = {}, {}
    for m in modalities:
        Ds_tr_norm[m], scales[m] = _normalize_distance_train(Ds_tr[m])

    Ds_te_tr_norm = {
        m: _apply_scale_to_test_block(Ds_te_tr[m], scales[m])
        for m in modalities
    }

    # 6. Gate features
    Phi_tr_full, mod2cols = _gate_features_block_scale_free(
        Ds_tr_norm,
        modalities,
        y_ref=y_train,
        k=8,
        is_square=True,
        class_list=conditions,
    )
    Phi_te_full, _ = _gate_features_block_scale_free(
        Ds_te_tr_norm,
        modalities,
        y_ref=y_train,
        k=8,
        is_square=False,
        class_list=conditions,
    )

    # 7. Gate supervision on TRAIN
    C_bin_tr, P_tr = _gate_multilabel_targets_from_KNN_LOO_train(
        Ds_tr_norm,
        modalities,
        y_train,
        k=8,
        tau=0.5,
        return_props=True,
    )
    sample_weights_list = [1e-2 + P_tr[:, mi] for mi in range(len(modalities))]

    _, gate_logits = _fit_gate_multilabel_scale_free(
        Phi_tr_full,
        C_bin_tr,
        mod2cols,
        C=0.5,
        max_iter=1000,
        class_weight="balanced",
        sample_weights_list=sample_weights_list,
    )

    # 8. Gate weights
    Z_tr = gate_logits(Phi_tr_full)
    Z_te = gate_logits(Phi_te_full)
    W_tr = _gate_weights_from_logits(Z_tr, temp=gate_temp, mix_uniform=0.0)
    W_te = _gate_weights_from_logits(Z_te, temp=gate_temp, mix_uniform=0.0)

    # 9. Kernel fusion
    Ks_tr, Ks_te_tr = _kernelize_blocks(
        Ds_tr_norm,
        Ds_te_tr_norm,
        use_global_sigma=False,
    )
    K_tr_fused = fuse_train_pairwise_kernels_weighted_mean(
        Ks_tr,
        modalities,
        W_tr,
    )
    K_te_tr_fused = fuse_test_pairwise_kernels_weighted_mean(
        Ks_te_tr,
        modalities,
        W_te,
        W_tr,
    )

    # 10. SVM with precomputed kernel
    clf = SVC(C=2.0, kernel="precomputed")
    clf.fit(K_tr_fused, y_train)
    y_pred = clf.predict(K_te_tr_fused)

    acc_fold = float(np.mean(y_pred == y_test))
    f1_fold = float(f1_score(y_test, y_pred, average="macro"))

    # Confusion contribution
    conf_mat = np.zeros((len(conditions), len(conditions)))
    for label_true, label_pred in zip(y_test, y_pred):
        conf_mat[
            conditions_dict[label_true],
            conditions_dict[label_pred],
        ] += 1

    # Mean gate weights over test instances
    w_te_mean = W_te.mean(axis=0) if W_te.size > 0 else np.zeros(len(modalities))

    return {
        "acc": acc_fold,
        "f1": f1_fold,
        "confmat": conf_mat,
        "w_te_mean": w_te_mean,
        "modalities": modalities,
    }


# --------------------------------------------------------------------
# Cross-validation pipeline
# --------------------------------------------------------------------


def run_crossval_pipeline(config, path, task):
    """
    High-level orchestration for LOSO:
      - load raw data
      - build labels and LOSO groups
      - parallelize over folds
      - aggregate metrics and plot confusion matrix
    """
    # 1) Load raw data
    raw_data = load_raw_features(config, path)

    # 2) Usable recordings (subject_label strings)
    all_records = sorted(
        {
            f"{subj}_{lab}"
            for (subj, lab, _) in raw_data.keys()
            if lab in [str(i) for i in range(1, 10)]
        }
    )

    # 3) Labels map 
    if task == "binary":
        dict_task = {
            "1": "low_wl",
            "2": "low_wl",
            "3": "low_wl",
            "4": "low_wl",
            "5": "high_wl",
            "6": "high_wl",
            "7": "high_wl",
            "8": "high_wl",
            "9": "high_wl",
        }
        conditions = ["low_wl", "high_wl"]
    elif task == "ternary":
        dict_task = {
            "1": "low_wl",
            "2": "low_wl",
            "3": "low_wl",
            "4": "medium_wl",
            "5": "medium_wl",
            "6": "medium_wl",
            "7": "high_wl",
            "8": "high_wl",
            "9": "high_wl",
        }
        conditions = ["low_wl", "medium_wl", "high_wl"]
    else:
        raise ValueError("task must be 'binary' or 'ternary'.")

    y_all = np.array([dict_task[r.split("_")[1]] for r in all_records])
    subjects = np.array([r.split("_")[0] for r in all_records])
    conditions_dict = {c: i for i, c in enumerate(conditions)}

    # 4) LOSO splitter
    logo = LeaveOneGroupOut()
    folds = list(logo.split(all_records, groups=subjects))
    print(f"Running LOSO with {len(folds)} folds | {task} task")

    # 5) Parallel execution of folds
    fold_results = Parallel(n_jobs=-1, backend="loky")(
        delayed(run_loso_fold)(
            train_index,
            test_index,
            config,
            raw_data,
            all_records,
            y_all,
            conditions,
            conditions_dict,
            task,
        )
        for train_index, test_index in folds
    )

    # 6) Aggregate metrics
    accs = [fr["acc"] for fr in fold_results]
    f1s = [fr["f1"] for fr in fold_results]
    conf_sum = np.sum([fr["confmat"] for fr in fold_results], axis=0)

    print(f"LOSO accuracy: mean={np.mean(accs):.4f} | std={np.std(accs):.4f}")
    print(f"LOSO F1 (macro): mean={np.mean(f1s):.4f} | std={np.std(f1s):.4f}")
    print("Confusion matrix (sum over folds):")
    print(conf_sum)

    # 6b) Mean gate weights across folds
    modalities = fold_results[0]["modalities"]
    w_mat = np.stack([fr["w_te_mean"] for fr in fold_results], axis=0)
    w_mean_across_folds = w_mat.mean(axis=0)

    print("\nMean gate weights across folds (averaged over all test instances):")
    for m, w in zip(modalities, w_mean_across_folds):
        print(f"  - {m:>20s}: {w:.4f}")
    print(f"\nSanity — sum of mean weights: {w_mean_across_folds.sum():.4f}")

    # 7) Plot confusion matrix
    disp = ConfusionMatrixDisplay(conf_sum.astype(int), display_labels=conditions)
    disp.plot(values_format="", colorbar=False, cmap="Blues")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run symbolic pipeline")
     
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Use the CLDrive dataset with binary task"
    )
    parser.add_argument(
        "--ternary",
        action="store_true",
        help="Use the CLDrive dataset with ternary task"
    )
   
    args = parser.parse_args()
     
    if args.binary:
        task = 'binary' 
    elif args.ternary:
        task = 'ternary' 
    else:
        raise ValueError("Please specify one task using --binary or --ternary")
     
        
    with open("configuration/analysis_cldrive.yaml", "r") as file:
        config = yaml.safe_load(file)

    path = "input/features/"
    run_crossval_pipeline(config, path, task)




