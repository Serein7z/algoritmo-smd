"""
Figuras de resultados (versión 2). Solo usa resultados ya guardados por smd_pipeline.py:
  resultados/scores_machine-1-1.npz   (etiquetas reales y predicciones de cada algoritmo)
  resultados/metricas_todas.csv       (métricas de las 28 máquinas)
No recalcula ni modifica ningún dato experimental.

Genera en figuras/:
  fig2_deteccion_machine-1-1.png   línea de tiempo: anomalías reales frente a alertas (TP y falsas alarmas)
  fig3_f1_por_maquina.png          distribución por máquina de F1 y F1-PA (caja + puntos + media)
  fig_extra_precision_recall.png   (propuesta) compromiso Precision-Recall en las 28 máquinas
"""
import os; os.makedirs("figuras", exist_ok=True)
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ALG = ["K-Means", "GMM", "DBSCAN"]
C = {"K-Means": "#1F5FA8", "GMM": "#C2620A", "DBSCAN": "#15996B"}
INK, MUTED, GRID, TRUTH = "#222222", "#666666", "#E3E3E0", "#4A4A4A"
plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False})
coma = lambda v, d=2: f"{v:.{d}f}".replace(".", ",")

# ------------------------------------------------------------------ Fig. 2
z = np.load("resultados/scores_machine-1-1.npz")
y = z["y"].astype(int); n = len(y); W = 200
nb = int(np.ceil(n / W)); xc = (np.arange(nb) + 0.5) * W
d = np.diff(np.r_[0, y, 0]); ini = np.where(d == 1)[0]; fin = np.where(d == -1)[0]

def por_ventana(v):
    pad = np.r_[v, np.zeros(nb * W - n)]
    cuenta = np.r_[np.ones(n), np.zeros(nb * W - n)].reshape(nb, W).sum(1)
    return 100 * pad.reshape(nb, W).sum(1) / cuenta

fig = plt.figure(figsize=(7.2, 3.75), dpi=300)
gs = fig.add_gridspec(4, 1, height_ratios=[0.55, 1, 1, 1], hspace=0.16, left=0.155, right=0.80, top=0.875, bottom=0.125)
axs = [fig.add_subplot(gs[i]) for i in range(4)]
for ax in axs[:-1]: ax.tick_params(labelbottom=False)
for ax in axs: ax.set_xlim(0, n)

ax = axs[0]
for a, b in zip(ini, fin): ax.axvspan(a, b, color=TRUTH, lw=0)
ax.set_yticks([]); ax.set_ylim(0, 1); ax.spines["left"].set_visible(False)
ax.set_ylabel("Anomalía\nreal", rotation=0, ha="right", va="center", fontsize=7.5, color=TRUTH, fontweight="bold")
ax.text(1.012, 0.5, f"{len(ini)} segmentos (3 muy cortos)\n{int(y.sum()):,}".replace(",", " ") + f" observaciones ({coma(100*y.mean(),2)} %)",
        transform=ax.transAxes, va="center", fontsize=6.6, color=TRUTH)

for ax, a in zip(axs[1:], ALG):
    p = z[f"{a}_pred"].astype(int)
    TP = int(((p == 1) & (y == 1)).sum()); FP = int(((p == 1) & (y == 0)).sum()); FN = int(((p == 0) & (y == 1)).sum())
    P, R = TP / (TP + FP), TP / (TP + FN); F1 = 2 * P * R / (P + R)
    for s, e in zip(ini, fin): ax.axvspan(s, e, color="#9A9A9A", alpha=.22, lw=0, zorder=0)
    tp = por_ventana(((p == 1) & (y == 1)).astype(float)); fp = por_ventana(((p == 1) & (y == 0)).astype(float))
    ax.fill_between(xc, 0, tp, step="mid", color=C[a], alpha=.95, lw=0, zorder=2)
    ax.fill_between(xc, tp, tp + fp, step="mid", color=C[a], alpha=.32, lw=0, zorder=2)
    ax.set_ylim(0, 100); ax.set_yticks([0, 50, 100]); ax.grid(axis="y", color=GRID, lw=.5, zorder=0)
    ax.set_ylabel(a, rotation=0, ha="right", va="center", fontsize=7.5, color=C[a], fontweight="bold", labelpad=6)
    ax.text(1.012, 0.5, f"Precision {coma(P)}\nRecall {coma(R)}\nF1 {coma(F1)}\nAlertas: {coma(100*p.mean(),1)} % de las obs.",
            transform=ax.transAxes, va="center", fontsize=6.6, color=INK, linespacing=1.3)
axs[-1].set_xlabel("Observación del conjunto de prueba de machine-1-1 (orden temporal)")
fig.text(0.014, 0.42, "Observaciones con alerta por ventana de 200 (%)", rotation=90, va="center", fontsize=6.8, color=MUTED)
leyenda = [Patch(fc="#9A9A9A", alpha=.4, label="Anomalía real (etiqueta)"),
           Patch(fc="#555555", alpha=.95, label="Alerta dentro de una anomalía real (acierto)"),
           Patch(fc="#555555", alpha=.32, label="Alerta fuera de las anomalías (falsa alarma)")]
fig.legend(handles=leyenda, loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.45, 0.995), handlelength=1.4, columnspacing=1.6)
fig.savefig("figuras/fig2_deteccion_machine-1-1.png"); plt.close()

# ------------------------------------------------------------------ Fig. 3
df = pd.read_csv("resultados/metricas_todas.csv")
rng = np.random.RandomState(0)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), dpi=300, sharey=True)
fig.subplots_adjust(left=0.085, right=0.985, top=0.895, bottom=0.235, wspace=0.07)
for ax, col, tit in zip(axs, ["f1", "f1_pa"], ["(a) F1-score estándar", "(b) F1-score con ajuste por puntos (F1-PA)"]):
    for i, a in enumerate(ALG):
        v = df[df.algoritmo == a][col].values
        bp = ax.boxplot(v, positions=[i], widths=.5, patch_artist=True, showfliers=False, manage_ticks=False,
                        boxprops=dict(facecolor=C[a], alpha=.20, edgecolor=C[a], lw=1.1),
                        whiskerprops=dict(color=C[a], lw=1.1), capprops=dict(color=C[a], lw=1.1),
                        medianprops=dict(color=INK, lw=1.8))
        ax.scatter(i + rng.uniform(-.16, .16, len(v)), v, s=9, color=C[a], alpha=.85, edgecolor="white", lw=.4, zorder=3)
        ax.scatter([i], [v.mean()], marker="D", s=26, color="white", edgecolor=INK, lw=1.0, zorder=5)
        ax.text(i + .30, np.median(v), coma(np.median(v)), va="center", fontsize=6.8, color=INK)
    ax.set_xticks(range(3)); ax.set_xticklabels(ALG, fontsize=7.8); ax.set_xlim(-.5, 2.75)
    ax.set_ylim(0, 1.02); ax.set_title(tit, fontsize=7.8, color=INK, loc="left"); ax.grid(axis="y", color=GRID, lw=.6, zorder=0)
axs[0].set_ylabel("F1 por máquina (n = 28)")
leg = [Patch(fc="#888", alpha=.25, ec="#888", label="Rango intercuartílico (Q1–Q3)"),
       Line2D([0], [0], color=INK, lw=1.8, label="Mediana (valor indicado)"),
       Line2D([0], [0], marker="D", color="w", markerfacecolor="white", markeredgecolor=INK, markersize=5, label="Media"),
       Line2D([0], [0], marker="o", color="w", markerfacecolor="#888", markersize=4.5, label="Una máquina")]
fig.legend(handles=leg, loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.54, 0.0))
fig.savefig("figuras/fig3_f1_por_maquina.png"); plt.close()

# ------------------------------------------------------------------ Propuesta: Precision vs Recall
fig, ax = plt.subplots(figsize=(7.2, 3.3), dpi=300)
fig.subplots_adjust(left=0.07, right=0.985, top=0.855, bottom=0.14)
rr = np.linspace(0.01, 1, 400)
for f in (0.2, 0.4, 0.6):
    pp = f * rr / (2 * rr - f); ok = (pp > 0) & (pp <= 1)
    ax.plot(rr[ok], pp[ok], color="#BBBBBB", lw=.8, ls="--", zorder=1)
    ax.text(0.995, f * 0.995 / (2 * 0.995 - f) + 0.012, f"F1 = {coma(f,1)}", fontsize=6.3, color="#999999", ha="right")
for a in ALG:
    s = df[df.algoritmo == a]
    ax.scatter(s.recall, s.precision, s=16, color=C[a], alpha=.55, edgecolor="white", lw=.4, zorder=3)
for a in ALG:
    s = df[df.algoritmo == a]; mx, my = s.recall.mean(), s.precision.mean()
    ax.scatter([mx], [my], s=75, color=C[a], edgecolor=INK, lw=1.2, marker="o", zorder=5)
ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02); ax.grid(color=GRID, lw=.5, zorder=0)
ax.set_xlabel("Recall (anomalías etiquetadas que se detectan)"); ax.set_ylabel("Precision (alertas que son anomalías reales)")
medias = {a: (df[df.algoritmo == a].recall.mean(), df[df.algoritmo == a].precision.mean()) for a in ALG}
hs = [Line2D([0], [0], marker="o", color="w", markerfacecolor=C[a], markeredgecolor=INK, markeredgewidth=1.1, markersize=8,
             label=f"{a}: media R {coma(medias[a][0])}, P {coma(medias[a][1])}") for a in ALG]
hs += [Line2D([0], [0], marker="o", color="w", markerfacecolor="#888", alpha=.6, markersize=5, label="Cada punto pequeño es una máquina"),
       Line2D([0], [0], color="#BBBBBB", ls="--", lw=.9, label="Curvas de F1 constante")]
ax.legend(handles=hs, loc="lower center", ncol=3, frameon=False, fontsize=6.9, bbox_to_anchor=(0.5, 1.0), columnspacing=1.6, handletextpad=0.5)
fig.savefig("figuras/fig_extra_precision_recall.png"); plt.close()
print("figuras generadas")