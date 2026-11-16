"""
Pipeline experimental: comparación de K-Means, GMM y DBSCAN para detección de
anomalías en el Server Machine Dataset (SMD).

Protocolo (CRISP-DM adaptado):
  1. Cada máquina se procesa por separado (como indica el README de SMD).
  2. Se conserva la partición original train/test (NO se mezclan filas: hay orden temporal).
  3. StandardScaler y PCA se AJUSTAN solo con train y se APLICAN a test.
  4. Los tres algoritmos usan la MISMA representación (PCA, varianza acumulada >= 95 %).
  5. Los hiperparámetros se eligen SOLO con datos de train (sin etiquetas):
       K-Means -> k por silhouette | GMM -> nº de componentes por BIC | DBSCAN -> eps por k-distancia.
  6. Umbral de decisión (K-Means, GMM): percentil Q de los puntajes de train (sin mirar el test).
     DBSCAN: un punto de test es anomalía si está a más de eps de todo punto núcleo.
  7. Las etiquetas de test se usan ÚNICAMENTE para calcular las métricas.

Uso:
  python smd_pipeline.py --data RUTA/ServerMachineDataset --machines machine-1-1
  python smd_pipeline.py --data RUTA/ServerMachineDataset --all
"""
import argparse, time, tracemalloc, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score, f1_score, silhouette_score

SEED = 42
Q_UMBRAL = 99.0          # percentil de train usado como umbral (K-Means, GMM)
VAR_PCA = 0.95           # varianza acumulada que conserva PCA
MAX_FIT_DBSCAN = None    # None = usa todo train


# ------------------------------------------------------------------ datos
def cargar(data_dir, nombre):
    d = Path(data_dir)
    f = nombre if nombre.endswith(".txt") else nombre + ".txt"
    for sub in ("train", "test", "test_label"):
        if not (d / sub / f).is_file():
            raise FileNotFoundError(d / sub / f)
    Xtr = pd.read_csv(d / "train" / f, header=None).to_numpy(np.float64)
    Xte = pd.read_csv(d / "test" / f, header=None).to_numpy(np.float64)
    y = pd.read_csv(d / "test_label" / f, header=None).iloc[:, 0].to_numpy(int)
    assert Xtr.shape[1] == Xte.shape[1] == 38 and len(y) == len(Xte)
    assert np.isfinite(Xtr).all() and np.isfinite(Xte).all()
    return Xtr, Xte, y


def preparar(Xtr, Xte):
    sc = StandardScaler().fit(Xtr)
    pca = PCA(n_components=VAR_PCA, random_state=SEED).fit(sc.transform(Xtr))
    return pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xte)), pca.n_components_


# ------------------------------------------------------------------ métricas
def ajuste_por_puntos(y, pred):
    """Point-adjustment: si se detecta algo dentro de un segmento anómalo, todo el segmento cuenta como detectado."""
    pred = pred.copy()
    n, i = len(y), 0
    while i < n:
        if y[i] == 1:
            j = i
            while j < n and y[j] == 1:
                j += 1
            if pred[i:j].any():
                pred[i:j] = 1
            i = j
        else:
            i += 1
    return pred


def metricas(y, pred, score=None):
    m = dict(
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0),
        f1=f1_score(y, pred, zero_division=0),
    )
    pa = ajuste_por_puntos(y, pred)
    m["f1_pa"] = f1_score(y, pa, zero_division=0)
    if score is not None:  # techo "oracle" (usa etiquetas de test): SOLO referencial
        ths = np.unique(np.quantile(score, np.linspace(0.80, 0.999, 120)))
        m["f1_oracle"] = max(f1_score(y, (score > t).astype(int), zero_division=0) for t in ths)
    return m


# ------------------------------------------------------------------ algoritmos
def run_kmeans(Ztr, Zte):
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(Ztr), min(5000, len(Ztr)), replace=False)
    best_k, best_s = 2, -1
    for k in range(2, 9):
        lab = KMeans(k, n_init=5, random_state=SEED).fit_predict(Ztr[idx])
        s = silhouette_score(Ztr[idx], lab)
        if s > best_s:
            best_k, best_s = k, s
    tracemalloc.start()
    t0 = time.perf_counter()
    km = KMeans(best_k, n_init=10, random_state=SEED).fit(Ztr)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    s_tr = km.transform(Ztr).min(axis=1)
    s_te = km.transform(Zte).min(axis=1)
    th = np.percentile(s_tr, Q_UMBRAL)
    t_pred = time.perf_counter() - t0
    pico = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
    return s_te, (s_te > th).astype(int), dict(k=int(best_k), _pico=pico), t_fit, t_pred


def run_gmm(Ztr, Zte):
    best, best_bic = None, np.inf
    for k in range(1, 7):
        g = GaussianMixture(k, covariance_type="full", random_state=SEED, reg_covar=1e-4).fit(Ztr)
        b = g.bic(Ztr)
        if b < best_bic:
            best, best_bic = k, b
    tracemalloc.start()
    t0 = time.perf_counter()
    gm = GaussianMixture(best, covariance_type="full", random_state=SEED, reg_covar=1e-4).fit(Ztr)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    s_tr = -gm.score_samples(Ztr)
    s_te = -gm.score_samples(Zte)
    th = np.percentile(s_tr, Q_UMBRAL)
    t_pred = time.perf_counter() - t0
    pico = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
    return s_te, (s_te > th).astype(int), dict(componentes=int(best), _pico=pico), t_fit, t_pred


def run_dbscan(Ztr, Zte):
    d = Ztr.shape[1]
    min_samples = max(2 * d, 10)
    nn = NearestNeighbors(n_neighbors=min_samples).fit(Ztr)
    kd = np.sort(nn.kneighbors(Ztr)[0][:, -1])
    eps = float(np.percentile(kd, 95))            # eps: percentil 95 de la k-distancia (solo train)
    tracemalloc.start()
    t0 = time.perf_counter()
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(Ztr)
    t_fit = time.perf_counter() - t0
    core = db.components_
    t0 = time.perf_counter()
    if len(core) == 0:
        s_te = np.full(len(Zte), np.inf)
    else:
        s_te = NearestNeighbors(n_neighbors=1).fit(core).kneighbors(Zte)[0][:, 0]
    t_pred = time.perf_counter() - t0
    pico = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
    info = dict(_pico=pico, eps=round(eps, 4), min_samples=int(min_samples),
                nucleos=int(len(core)), ruido_train=float((db.labels_ == -1).mean()))
    return s_te, (s_te > eps).astype(int), info, t_fit, t_pred


ALGOS = {"K-Means": run_kmeans, "GMM": run_gmm, "DBSCAN": run_dbscan}


def correr_maquina(data_dir, nombre, guardar=None):
    Xtr, Xte, y = cargar(data_dir, nombre)
    Ztr, Zte, ncomp = preparar(Xtr, Xte)
    filas, scores = [], {}
    for alg, fn in ALGOS.items():
        s, pred, info, t_fit, t_pred = fn(Ztr, Zte)
        pico = info.pop("_pico")
        m = metricas(y, pred, s if np.isfinite(s).all() else None)
        filas.append(dict(maquina=nombre, algoritmo=alg, n_train=len(Xtr), n_test=len(Xte),
                          pct_anomalias=round(100 * y.mean(), 3), pca_comp=int(ncomp),
                          **{k: round(v, 4) for k, v in m.items()},
                          pct_marcado=round(100 * pred.mean(), 3),
                          t_fit_s=round(t_fit, 3), t_pred_s=round(t_pred, 3),
                          mem_pico_MB=round(pico / 2**20, 1), hiperparametros=json.dumps(info)))
        scores[alg] = (s, pred)
    if guardar is not None:
        np.savez_compressed(guardar, y=y, **{f"{a}_score": v[0] for a, v in scores.items()},
                            **{f"{a}_pred": v[1] for a, v in scores.items()})
    return filas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--machines", nargs="*", default=["machine-1-1"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="resultados")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    maquinas = sorted(p.stem for p in (Path(a.data) / "train").glob("*.txt")) if a.all else a.machines
    todas = []
    for mname in maquinas:
        print(f">>> {mname}", flush=True)
        f = correr_maquina(a.data, mname, guardar=out / f"scores_{mname}.npz")
        todas += f
        print(pd.DataFrame(f)[["algoritmo", "precision", "recall", "f1", "f1_pa", "t_fit_s", "mem_pico_MB"]].to_string(index=False), flush=True)
    df = pd.DataFrame(todas)
    nombre = "metricas_todas.csv" if a.all else "metricas_" + "_".join(maquinas) + ".csv"
    df.to_csv(out / nombre, index=False)
    print("Guardado:", out / nombre)


if __name__ == "__main__":
    main()
