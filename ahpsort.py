"""
AHPSort II · Práctica 6 de Sistemas de Ayuda a la Decisión (Universidad de Jaén)

Funciones reutilizables para los cuatro ejercicios:

  * ahp_weights(matrix)          -> pesos de los criterios (método de autovalores), λmax, CI y CR.
  * PriorityFunction(x, y)       -> función de prioridad local lineal por tramos de un criterio,
                                    construida a partir de los puntos de referencia (RP) y de los
                                    perfiles centrales (CP) con sus prioridades.
  * AHPSortII                    -> agrega las prioridades locales con los pesos, calcula la
                                    prioridad global de los perfiles centrales y de las
                                    alternativas y asigna cada alternativa a la clase del
                                    perfil central más cercano.
  * plot_classification(...)     -> mapa de coropletas con la clase de cada provincia.

Notación (teoría AHPSort II, Miccoli e Ishizaka 2017):
    p_kj  = prioridad local de la alternativa k en el criterio j (interpolada en la función)
    p_k   = Σ_j w_j · p_kj                       (prioridad global de la alternativa k)
    cp_i  = Σ_j w_j · p_ij                       (prioridad global del perfil central i)
    Asignación con perfiles centrales: la alternativa va a la clase cuyo cp_i está más cerca de p_k.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

# Índice aleatorio de Saaty (n = 1..10)
RANDOM_INDEX = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24,
                7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
CR_THRESHOLD = 0.10

CRITERIOS = ["Crímenes", "Robos", "Vehículos", "Orden público", "Drogas"]
CLASES = ["Alto", "Medio", "Bajo"]          # nivel de SEGURIDAD (CP1, CP2, CP3)
COLORES_CLASE = {"Alto": "#2E7D32", "Medio": "#F9A825", "Bajo": "#C62828"}


# --------------------------------------------------------------------------- #
#  1. Pesos de los criterios (AHP, método de autovalores)
# --------------------------------------------------------------------------- #
@dataclass
class AHPResult:
    weights: np.ndarray
    lambda_max: float
    ci: float
    ri: float
    cr: float

    @property
    def is_consistent(self) -> bool:
        return self.cr < CR_THRESHOLD

    def table(self, names: Sequence[str]) -> pd.DataFrame:
        return pd.DataFrame({"Peso (w)": self.weights}, index=list(names))


def ahp_weights(matrix) -> AHPResult:
    """Pesos = autovector principal normalizado de la matriz de comparación a pares.
    CI = (λmax − n)/(n − 1); CR = CI / RI(n)."""
    A = np.asarray(matrix, dtype=float)
    n = A.shape[0]
    values, vectors = np.linalg.eig(A)
    k = int(np.argmax(values.real))
    lam = float(values[k].real)
    v = np.abs(vectors[:, k].real)
    w = v / v.sum()
    ci = (lam - n) / (n - 1) if n > 1 else 0.0
    ri = RANDOM_INDEX.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0.0
    return AHPResult(w, lam, ci, ri, cr)


# --------------------------------------------------------------------------- #
#  2. Función de prioridad local (lineal por tramos)
# --------------------------------------------------------------------------- #
class PriorityFunction:
    """Función de prioridad local de un criterio a partir de los puntos (x, y):
    x = valor del criterio (RP o CP), y = prioridad local obtenida por comparación a pares.

    Para un valor g entre dos puntos consecutivos (s_o, p_o) y (s_o+1, p_o+1):
        p(g) = p_o + (p_o+1 − p_o) / (s_o+1 − s_o) · (g − s_o)        (interpolación lineal)
    Fuera del rango [RP1, RP6] se TRUNCA: se asigna la prioridad del extremo correspondiente.
    """

    def __init__(self, x: Sequence[float], y: Sequence[float], name: str = ""):
        order = np.argsort(np.asarray(x, dtype=float), kind="stable")
        self.x = np.asarray(x, dtype=float)[order]
        self.y = np.asarray(y, dtype=float)[order]
        self.name = name
        if np.any(np.diff(self.x) <= 0):
            raise ValueError(f"Puntos repetidos en la función de «{name}»")

    def __call__(self, g):
        # np.interp interpola linealmente y trunca fuera del rango (usa y[0] / y[-1])
        return np.interp(np.asarray(g, dtype=float), self.x, self.y)

    def interpolation_detail(self, g: float) -> str:
        """Texto con el cálculo de la interpolación (útil para la memoria)."""
        if g <= self.x[0]:
            return f"{g:g} ≤ {self.x[0]:g} → truncado: p = {self.y[0]:.4f}"
        if g >= self.x[-1]:
            return f"{g:g} ≥ {self.x[-1]:g} → truncado: p = {self.y[-1]:.4f}"
        o = int(np.searchsorted(self.x, g, side="right") - 1)
        x0, x1, y0, y1 = self.x[o], self.x[o + 1], self.y[o], self.y[o + 1]
        p = y0 + (y1 - y0) / (x1 - x0) * (g - x0)
        return (f"{g:g} ∈ [{x0:g}, {x1:g}] → p = {y0:.4f} + ({y1:.4f} − {y0:.4f})/"
                f"({x1:g} − {x0:g})·({g:g} − {x0:g}) = {p:.4f}")


# --------------------------------------------------------------------------- #
#  3. AHPSort II
# --------------------------------------------------------------------------- #
class AHPSortII:
    def __init__(self, criteria: List[str], weights: Sequence[float],
                 functions: Dict[str, PriorityFunction],
                 central_profiles: Dict[str, Sequence[float]],
                 classes: List[str] = CLASES):
        self.criteria = criteria
        self.w = np.asarray(weights, dtype=float)
        self.functions = functions
        self.central_profiles = central_profiles     # criterio -> [CP1, CP2, CP3]
        self.classes = classes

    # --- perfiles centrales -------------------------------------------------
    def profiles_local(self) -> pd.DataFrame:
        return pd.DataFrame({c: self.functions[c](self.central_profiles[c]) for c in self.criteria},
                            index=self.classes)

    def profiles_table(self) -> pd.DataFrame:
        t = self.profiles_local()
        t["Global"] = t[self.criteria].values @ self.w
        return t

    # --- alternativas -------------------------------------------------------
    def local_priorities(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({c: self.functions[c](data[c].values) for c in self.criteria},
                            index=data.index)

    def classify(self, data: pd.DataFrame) -> pd.DataFrame:
        cp = self.profiles_table()["Global"].values
        t = self.local_priorities(data)
        t["Global"] = t[self.criteria].values @ self.w
        dist = np.abs(t["Global"].values[:, None] - cp[None, :])
        for i, cls in enumerate(self.classes):
            t[f"|p − cp{i + 1}|"] = dist[:, i]
        t["Nivel de seguridad"] = [self.classes[k] for k in dist.argmin(axis=1)]
        # valores fuera del rango de la función (truncados)
        # (tolerancia del 0,1 % para no marcar valores que solo superan el extremo por redondeo,
        #  p. ej. Barcelona 406,464 robos frente al RP6 = 406,46 redondeado)
        def fuera(c, v):
            lo, hi = self.functions[c].x[0], self.functions[c].x[-1]
            tol = 1e-3 * (hi - lo)
            return v < lo - tol or v > hi + tol
        out = [", ".join(c for c in self.criteria if fuera(c, data.loc[idx, c]))
               for idx in data.index]
        if any(out):
            t["Truncado en"] = out
        return t


# --------------------------------------------------------------------------- #
#  4. Utilidades de datos
# --------------------------------------------------------------------------- #
def strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", str(s)) if unicodedata.category(ch) != "Mn")


#: Código INE de provincia para cada nombre que aparece en los ficheros de datos.
INE = {
    "alava": "01", "araba": "01", "araba/alava": "01", "albacete": "02", "alicante": "03",
    "alicante/alacant": "03", "almeria": "04", "avila": "05", "badajoz": "06",
    "baleares": "07", "balears (illes)": "07", "balears, illes": "07", "barcelona": "08",
    "burgos": "09", "caceres": "10", "cadiz": "11", "castellon": "12", "castellon/castello": "12",
    "ciudad real": "13", "cordoba": "14", "coruna": "15", "coruna (a)": "15", "coruna, a": "15",
    "cuenca": "16", "girona": "17", "granada": "18", "guadalajara": "19", "gipuzkoa": "20",
    "huelva": "21", "huesca": "22", "jaen": "23", "leon": "24", "lleida": "25", "rioja": "26",
    "rioja (la)": "26", "rioja, la": "26", "lugo": "27", "madrid": "28", "malaga": "29",
    "murcia": "30", "navarra": "31", "ourense": "32", "asturias": "33", "palencia": "34",
    "palmas": "35", "palmas (las)": "35", "palmas, las": "35", "pontevedra": "36",
    "salamanca": "37", "santa cruz": "38", "santa cruz de tenerife": "38", "cantabria": "39",
    "segovia": "40", "sevilla": "41", "soria": "42", "tarragona": "43", "teruel": "44",
    "toledo": "45", "valencia": "46", "valencia/valencia": "46", "valladolid": "47",
    "bizkaia": "48", "zamora": "49", "zaragoza": "50", "ceuta": "51", "melilla": "52",
}


def ine_code(name: str) -> str:
    key = strip_accents(name).strip().lower()
    if key not in INE:
        raise KeyError(f"Provincia no reconocida: {name}")
    return INE[key]


def read_andalucia(path: str) -> pd.DataFrame:
    """Lee Andalucía.xls (columnas resumen de la derecha: Crímenes, Robos, Vehículos,
    Orden Público, Drogas) y devuelve una fila por provincia con los hechos absolutos."""
    raw = pd.read_excel(path, header=None)
    rows, prov = {}, None
    label_map = {"crimenes": "Crímenes", "robos": "Robos", "relaciondo con vehiculos": "Vehículos",
                 "relacionado con vehiculos": "Vehículos", "orden publico": "Orden público",
                 "drogas": "Drogas"}
    for _, r in raw.iterrows():
        a, b, c, d = r[0], r[1], r[2], r[3]
        if isinstance(a, str) and pd.isna(b) and pd.isna(c) and strip_accents(a).strip().lower() in INE:
            prov = a.strip()
            rows[prov] = {}
        if prov and isinstance(c, str):
            key = strip_accents(c).strip().lower()
            if key in label_map:
                rows[prov][label_map[key]] = float(d)
    df = pd.DataFrame.from_dict(rows, orient="index")[CRITERIOS]
    df.index.name = "Provincia"
    return df


# --------------------------------------------------------------------------- #
#  5. Mapas
# --------------------------------------------------------------------------- #
def load_map(path: str, kind: str):
    """Carga el shapefile y añade la columna 'INE' (código de provincia)."""
    import geopandas as gpd
    g = gpd.read_file(path)
    if kind == "andalucia":
        g["INE"] = g["COD_PROV"].astype(str).str.zfill(2)
        g["Nombre"] = g["PROVINCIA"]
    else:  # España (IGN, recintos provinciales): NATCODE = 34 + CCAA(2) + PROV(2) + ...
        g["INE"] = g["NATCODE"].astype(str).str[4:6]
        g["Nombre"] = g["NAMEUNIT"]
        g = g[~g["INE"].isin(["51", "52"])]          # sin Ceuta y Melilla
    return g


def plot_classification(gdf, result: pd.DataFrame, title: str, ax=None, labels=True,
                        class_col: str = "Nivel de seguridad"):
    """Mapa de coropletas con leyenda fija (Alto / Medio / Bajo) para poder comparar mapas."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    res = result.copy()
    res["INE"] = [ine_code(n) for n in res.index]
    m = gdf.merge(res[["INE", class_col]], on="INE", how="left")
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 7))
    colors = m[class_col].map(COLORES_CLASE).fillna("#DDDDDD")
    m.plot(color=colors, edgecolor="white", linewidth=0.8, ax=ax)
    if labels:
        for _, row in m.iterrows():
            pt = row.geometry.representative_point()
            ax.annotate(row["Nombre"], (pt.x, pt.y), ha="center", fontsize=7 if len(m) > 10 else 9,
                        color="black")
    handles = [Patch(color=COLORES_CLASE[c], label=c) for c in CLASES]
    if m[class_col].isna().any():
        handles.append(Patch(color="#DDDDDD", label="Sin datos"))
    ax.legend(handles=handles, loc="lower right", title="Nivel de seguridad", frameon=True)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_axis_off()
    return ax


def plot_function(f: PriorityFunction, cps: Sequence[float], ax=None, title: str = ""):
    """Dibuja una función de prioridad local con los RP (azul) y los CP (rojo)."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(f.x, f.y, "-", color="#2E75B6", lw=1.5)
    is_cp = np.isin(np.round(f.x, 6), np.round(np.asarray(cps, float), 6))
    ax.scatter(f.x[~is_cp], f.y[~is_cp], color="#2E75B6", zorder=3, label="RP")
    ax.scatter(f.x[is_cp], f.y[is_cp], color="#C62828", zorder=4, label="CP")
    ax.set_title(title or f.name, fontsize=10, fontweight="bold")
    ax.set_xlabel("Valor del criterio")
    ax.set_ylabel("Prioridad local")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return ax


# --------------------------------------------------------------------------- #
#  6. Lectura y normalización de datos (ejercicios 2-4)
# --------------------------------------------------------------------------- #
def read_habitantes(path: str) -> pd.Series:
    """Población por provincia (fichero habitantes.xlsx), indexada por código INE."""
    raw = pd.read_excel(path, header=None)
    rows = raw.dropna(subset=[1, 2])
    rows = rows[rows[2].apply(lambda v: isinstance(v, (int, float)))]
    return pd.Series(rows[2].astype(float).values, index=[ine_code(n) for n in rows[1]],
                     name="Habitantes")


def per_10000(absolute: pd.DataFrame, population: pd.Series) -> pd.DataFrame:
    """Tasa = hechos / habitantes × 10.000 (población buscada por código INE)."""
    pop = pd.Series([population[ine_code(p)] for p in absolute.index], index=absolute.index)
    return absolute.div(pop, axis=0) * 10000


def read_nacional_2017(path: str) -> pd.DataFrame:
    df = pd.read_excel(path).set_index("Provincia")
    df.columns = CRITERIOS
    return df


def read_nacional_2023(path: str) -> pd.DataFrame:
    df = pd.read_excel(path).set_index("Provincia")
    rates = df[["Crímenes", "Robos", "Vehículos", "Orden público", "Drogas"]].copy()
    rates.columns = CRITERIOS
    return rates


def derive_profiles(rates: pd.DataFrame, percentiles=(0.2, 0.5, 0.8), n_rp: int = 6):
    """Cómo se obtienen los CP y RP a partir de los datos:
       CP1, CP2, CP3 = percentiles 20, 50 y 80 de las tasas (seguridad alta, media y baja);
       RP1..RP6 = 6 puntos equiespaciados entre 0 y el máximo observado."""
    cps = rates.quantile(list(percentiles))
    cps.index = ["CP1 (Alto)", "CP2 (Medio)", "CP3 (Bajo)"]
    rps = pd.DataFrame({c: np.linspace(0, rates[c].max(), n_rp) for c in rates.columns},
                       index=[f"RP{k}" for k in range(1, n_rp + 1)])
    return cps, rps
