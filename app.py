import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# AYEDAŞ | SCADA Demand + CBS Kurulu Güç Kapasite Paneli
# =========================================================
# Repo kök dizinine şu dosyaları koy:
# - app.py
# - requirements.txt
# - Sancaktepe Trafo demand 2025 yeni.xlsx  veya  Sancaktepe Trafo demand 2025.xlsx
# - Trafo Sorgu Sonuçları.xlsx
# =========================================================

SCADA_FILE_CANDIDATES = [
    "Sancaktepe Trafo demand 2025 yeni.xlsx",
    "Sancaktepe Trafo demand 2025.xlsx",
]
CBS_FILE_CANDIDATES = [
    "Trafo Sorgu Sonuçları.xlsx",
    "Trafo Sorgu Sonuclari.xlsx",
]


# -----------------------------
# Genel yardımcılar
# -----------------------------
def find_existing_file(candidates: list[str]) -> Optional[Path]:
    """Repo içinde aday dosya adlarından var olan ilkini bulur."""
    for name in candidates:
        p = Path(name)
        if p.exists():
            return p
    return None


def normalize_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_key(x) -> str:
    """
    T-4092, T4092, t 4092 gibi değerleri aynı anahtara indirir.
    Sadece harf/rakam bırakır ve büyük harfe çevirir.
    """
    s = normalize_text(x).upper()
    return re.sub(r"[^A-Z0-9]", "", s)


def normalize_tr_code(x) -> str:
    """TR1, tr-1, TR 1 gibi değerleri normalize eder."""
    s = normalize_text(x).upper()
    s = re.sub(r"\s+", "", s)
    return s


def is_yes(x) -> bool:
    s = normalize_text(x).lower()
    return s in {"evet", "e", "yes", "y", "true", "1", "var"}


def is_quality_valid(q: str) -> bool:
    """
    'invalid' içinde de 'valid' geçtiği için contains('valid') tek başına hatalıdır.
    """
    s = normalize_text(q).lower()
    return ("valid" in s) and ("invalid" not in s)


def fmt_num(x, digits=2, suffix=""):
    if pd.isna(x) or not np.isfinite(float(x)):
        return "-"
    return f"{float(x):,.{digits}f}{suffix}"


# -----------------------------
# SCADA okuma + ayrıştırma
# -----------------------------
@st.cache_data(show_spinner=False)
def load_scada_excel(file_path: str) -> pd.DataFrame:
    """
    SCADA export'ta header kaymış olabiliyor.
    header=None okuyup 'Point Name' ve 'Time stamp' satırını header yapar.
    """
    raw = pd.read_excel(file_path, sheet_name=0, header=None)

    header_row_idx = None
    for i in range(min(20, len(raw))):
        row = raw.iloc[i].astype(str).str.lower().tolist()
        if any("point name" in c for c in row) and any("time stamp" in c for c in row):
            header_row_idx = i
            break

    if header_row_idx is None:
        header_row_idx = 0

    header = raw.iloc[header_row_idx].tolist()
    df = raw.iloc[header_row_idx + 1 :].copy()
    df.columns = header

    col_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if "point name" in cl:
            col_map[c] = "point_name"
        elif "time stamp" in cl:
            col_map[c] = "timestamp"
        elif "millisecond" in cl:
            col_map[c] = "ms"
        elif cl == "value" or " value" in cl:
            col_map[c] = "value"
        elif "source" in cl or "quality" in cl:
            col_map[c] = "quality"

    df = df.rename(columns=col_map)

    needed = {"point_name", "timestamp", "value"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(
            f"SCADA Excel formatında beklenen kolonlar bulunamadı: {missing}. "
            f"Mevcut kolonlar: {list(df.columns)}"
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    if "quality" not in df.columns:
        df["quality"] = "Unknown"

    df = df.dropna(subset=["point_name", "timestamp", "value"])
    df["point_name"] = df["point_name"].astype(str)

    return df[["point_name", "timestamp", "value", "quality"]]


def extract_scada_keys(point_name: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Örnek:
    /Net-E/SANCAKTEPE OM DTM/T-4092/0.4kV/Enan2 H01/S

    dm_id          = T-4092
    montaj_key     = T4092     -> CBS Montaj Yeri ile eşleşir
    enan_no        = 2
    tr_code_norm   = TR2       -> CBS Trafo Kodu ile eşleşir
    h_cell         = H01       -> hücre/nokta detayı olarak tutulur
    """
    p = normalize_text(point_name)

    dm_id = None
    m_dm = re.search(r"/(T-?\d+)", p, flags=re.IGNORECASE)
    if m_dm:
        raw_dm = m_dm.group(1).upper()
        raw_dm = raw_dm.replace("T", "T-") if raw_dm.startswith("T") and "-" not in raw_dm else raw_dm
        dm_id = raw_dm

    enan_no = None
    m_enan = re.search(r"\bEnan\s*[-_ ]?(\d+)\b|\bEnan(\d+)\b", p, flags=re.IGNORECASE)
    if m_enan:
        enan_no = next((g for g in m_enan.groups() if g), None)

    tr_code_norm = f"TR{enan_no}" if enan_no else None

    h_cell = None
    m_h = re.search(r"\bH\s*[-_ ]?(\d{1,3})\b|\bH(\d{2})\b", p, flags=re.IGNORECASE)
    if m_h:
        h_no = next((g for g in m_h.groups() if g), None)
        if h_no:
            h_cell = f"H{int(h_no):02d}"

    metric = None
    m_metric = re.search(r"/([A-Za-z0-9]+)\s*$", p.strip())
    if m_metric:
        metric = m_metric.group(1).upper()

    montaj_key = normalize_key(dm_id)
    return dm_id, montaj_key, enan_no, tr_code_norm, h_cell, metric


def prepare_scada(df: pd.DataFrame, only_valid: bool, remove_zeros: bool) -> pd.DataFrame:
    d = df.copy()

    parsed = d["point_name"].apply(extract_scada_keys)
    d["dm_id"] = parsed.apply(lambda x: x[0])
    d["montaj_key"] = parsed.apply(lambda x: x[1])
    d["enan_no"] = parsed.apply(lambda x: x[2])
    d["tr_code_norm"] = parsed.apply(lambda x: x[3])
    d["h_cell"] = parsed.apply(lambda x: x[4])
    d["metric"] = parsed.apply(lambda x: x[5])

    d = d.dropna(subset=["montaj_key", "tr_code_norm", "timestamp", "value"])

    if only_valid:
        d = d[d["quality"].apply(is_quality_valid)]

    if remove_zeros:
        d = d[d["value"] > 0]

    # Bu panel S görünür gücü / demand üzerinden ilerler.
    if "metric" in d.columns:
        d = d[d["metric"].eq("S") | d["metric"].isna()]

    return d


def aggregate_scada(d: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Aynı trafo aynı zaman kovasında birden çok kayıt varsa demand için MAX kullanılır.
    15 dakikalık veri varsa 15min, günlük/saatlik export varsa seçilen kovaya göre gruplanır.
    """
    if d.empty:
        return pd.DataFrame(
            columns=["montaj_key", "tr_code_norm", "timestamp", "demand_kva", "sample_count", "h_cell"]
        )

    x = d.copy()
    x["period"] = x["timestamp"].dt.floor(freq)

    g = (
        x.groupby(["montaj_key", "tr_code_norm", "period"], as_index=False)
        .agg(
            demand_kva=("value", "max"),
            sample_count=("value", "size"),
            h_cell=("h_cell", lambda s: ", ".join(sorted(set(s.dropna().astype(str))))),
            dm_id=("dm_id", "first"),
        )
        .rename(columns={"period": "timestamp"})
        .sort_values(["montaj_key", "tr_code_norm", "timestamp"])
    )
    return g


# -----------------------------
# CBS okuma + filtreleme
# -----------------------------
@st.cache_data(show_spinner=False)
def load_cbs_excel(file_path: str) -> pd.DataFrame:
    cbs = pd.read_excel(file_path, sheet_name=0)

    required = ["Montaj Yeri", "Gücü[kVA]", "Trafo Kodu", "SCADA-RTU Var mı?"]
    missing = [c for c in required if c not in cbs.columns]
    if missing:
        raise ValueError(
            f"CBS Excel formatında beklenen kolonlar bulunamadı: {missing}. "
            f"Mevcut kolonlar: {list(cbs.columns)}"
        )

    out = cbs.copy()
    out["montaj_yeri"] = out["Montaj Yeri"].apply(normalize_text)
    out["montaj_key"] = out["Montaj Yeri"].apply(normalize_key)
    out["tr_code_norm"] = out["Trafo Kodu"].apply(normalize_tr_code)
    out["kurulu_guc_kva"] = pd.to_numeric(out["Gücü[kVA]"], errors="coerce")
    out["scada_rtu_var"] = out["SCADA-RTU Var mı?"].apply(is_yes)

    # Kullanışlı kolonları standart adlarla ekle
    out["asset_id"] = out["AssetID"] if "AssetID" in out.columns else np.nan
    out["ilce"] = out["İlçe"] if "İlçe" in out.columns else ""
    out["mahalle"] = out["Mahalle"] if "Mahalle" in out.columns else ""
    out["marka"] = out["Marka"] if "Marka" in out.columns else ""
    out["tipi"] = out["Tipi"] if "Tipi" in out.columns else ""
    out["trafo_osos_var"] = out["Trafo-OSOS Var mı?"].apply(is_yes) if "Trafo-OSOS Var mı?" in out.columns else False

    out = out.dropna(subset=["montaj_key", "tr_code_norm", "kurulu_guc_kva"])
    out = out[out["montaj_key"].ne("") & out["tr_code_norm"].ne("")]

    return out


# -----------------------------
# Kapasite analizi
# -----------------------------
def estimate_period_hours(freq: str) -> float:
    try:
        td = pd.to_timedelta(freq)
        return max(td.total_seconds() / 3600, 1 / 60)
    except Exception:
        if freq.lower() in {"h", "1h", "hour", "saat"}:
            return 1.0
        return 0.25


def scada_metrics(scada_agg: pd.DataFrame, thermal_window_points: int) -> pd.DataFrame:
    if scada_agg.empty:
        return pd.DataFrame()

    rows = []
    for (montaj_key, tr_code_norm), g in scada_agg.groupby(["montaj_key", "tr_code_norm"]):
        y = g.sort_values("timestamp").copy()
        demand = y["demand_kva"].astype(float)

        # Kısa süreli sıçramalara karşı merkezli median. Asıl pikler ayrıca max/P99'da tutulur.
        filtered = demand.rolling(window=3, min_periods=1, center=True).median()
        thermal = filtered.rolling(window=thermal_window_points, min_periods=1).mean()

        rows.append(
            {
                "montaj_key": montaj_key,
                "tr_code_norm": tr_code_norm,
                "dm_id": y["dm_id"].dropna().iloc[-1] if y["dm_id"].notna().any() else montaj_key,
                "h_cell": ", ".join(sorted(set(", ".join(y["h_cell"].dropna().astype(str)).split(", ")) - {""})),
                "veri_adedi": int(len(y)),
                "ilk_veri": y["timestamp"].min(),
                "son_veri": y["timestamp"].max(),
                "son_demand_kva": float(demand.iloc[-1]) if len(demand) else np.nan,
                "maks_demand_kva": float(demand.max()) if len(demand) else np.nan,
                "ortalama_demand_kva": float(demand.mean()) if len(demand) else np.nan,
                "median_demand_kva": float(demand.median()) if len(demand) else np.nan,
                "p95_demand_kva": float(demand.quantile(0.95)) if len(demand) else np.nan,
                "p99_demand_kva": float(demand.quantile(0.99)) if len(demand) else np.nan,
                "termal_demand_kva": float(thermal.max()) if len(thermal) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def classify_load(load_pct: float, warn_pct: float, critical_pct: float) -> str:
    if pd.isna(load_pct):
        return "Veri Yok"
    if load_pct >= 100:
        return "Aşırı Yüklü"
    if load_pct >= critical_pct:
        return "Kritik"
    if load_pct >= warn_pct:
        return "Sınırda"
    return "Normal"


def build_capacity_analysis(
    cbs: pd.DataFrame,
    scada_agg: pd.DataFrame,
    only_cbs_scada: bool,
    planning_metric: str,
    thermal_window_points: int,
    warn_pct: float,
    critical_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Dönüş:
    - analysis: CBS + SCADA eşleşmiş kapasite metrikleri
    - cbs_without_scada_data: CBS'de SCADA-RTU Var ama SCADA demand gelmeyenler
    - scada_without_cbs: SCADA demand var ama CBS'de karşılığı bulunmayanlar
    """
    c = cbs.copy()
    if only_cbs_scada:
        c = c[c["scada_rtu_var"]]

    # Aynı Montaj Yeri + Trafo Kodu birden çok gelirse en güncel/ilk kayıt tekilleştirilir.
    c = c.sort_values(["montaj_key", "tr_code_norm"]).drop_duplicates(
        subset=["montaj_key", "tr_code_norm"], keep="first"
    )

    m = scada_metrics(scada_agg, thermal_window_points=thermal_window_points)

    join_cols = ["montaj_key", "tr_code_norm"]
    analysis = c.merge(m, on=join_cols, how="left", suffixes=("_cbs", "_scada"))

    if m.empty:
        scada_without_cbs = pd.DataFrame(columns=join_cols)
    else:
        scada_without_cbs = m.merge(c[join_cols], on=join_cols, how="left", indicator=True)
        scada_without_cbs = scada_without_cbs[scada_without_cbs["_merge"].eq("left_only")].drop(columns=["_merge"])

    cbs_without_scada_data = analysis[analysis["veri_adedi"].isna()].copy()

    metric_map = {
        "P95 demand": "p95_demand_kva",
        "P99 demand": "p99_demand_kva",
        "Termal demand": "termal_demand_kva",
        "Maks demand": "maks_demand_kva",
        "Ortalama demand": "ortalama_demand_kva",
    }
    chosen_col = metric_map.get(planning_metric, "p95_demand_kva")
    analysis["planlama_demand_kva"] = analysis[chosen_col]

    for col in [
        "son_demand_kva",
        "maks_demand_kva",
        "ortalama_demand_kva",
        "p95_demand_kva",
        "p99_demand_kva",
        "termal_demand_kva",
        "planlama_demand_kva",
    ]:
        analysis[f"{col}_yuklenme_pct"] = analysis[col] / analysis["kurulu_guc_kva"] * 100

    analysis["yuklenme_pct"] = analysis["planlama_demand_kva_yuklenme_pct"]
    analysis["bos_kapasite_kva_80"] = analysis["kurulu_guc_kva"] * (warn_pct / 100) - analysis["planlama_demand_kva"]
    analysis["bos_kapasite_kva_100"] = analysis["kurulu_guc_kva"] - analysis["planlama_demand_kva"]
    analysis["risk_durumu"] = analysis["yuklenme_pct"].apply(lambda x: classify_load(x, warn_pct, critical_pct))

    return analysis, cbs_without_scada_data, scada_without_cbs


def simulate_connection(analysis: pd.DataFrame, demand_kw: float, power_factor: float) -> pd.DataFrame:
    out = analysis.copy()
    pf = max(float(power_factor), 0.01)
    request_kva = float(demand_kw) / pf
    out["yeni_talep_kw"] = float(demand_kw)
    out["guc_faktoru"] = pf
    out["yeni_talep_kva"] = request_kva
    out["projeksiyon_demand_kva"] = out["planlama_demand_kva"] + request_kva
    out["projeksiyon_yuklenme_pct"] = out["projeksiyon_demand_kva"] / out["kurulu_guc_kva"] * 100
    out["baglanti_karari"] = out["projeksiyon_yuklenme_pct"].apply(lambda x: classify_connection(x))
    return out


def classify_connection(projected_pct: float) -> str:
    if pd.isna(projected_pct):
        return "Veri Yok"
    if projected_pct >= 100:
        return "Yatırım Gerekli"
    if projected_pct >= 95:
        return "Kritik İnceleme"
    if projected_pct >= 80:
        return "Şartlı Uygun / İzlenmeli"
    return "Uygun"


# -----------------------------
# Hamule ölçüm zamanı önerisi
# -----------------------------
def score_windows(g: pd.DataFrame, window_periods: int = 8) -> pd.DataFrame:
    g = g.copy().sort_values("timestamp")
    x = g["demand_kva"].astype(float)

    roll_mean = x.rolling(window_periods, min_periods=window_periods).mean()
    roll_std = x.rolling(window_periods, min_periods=window_periods).std()
    roll_diff = x.diff().abs().rolling(window_periods, min_periods=window_periods).mean()

    def z(v):
        arr = v.to_numpy(dtype=float)
        mu = np.nanmean(arr)
        sd = np.nanstd(arr) + 1e-9
        return (arr - mu) / sd

    mean_z = z(roll_mean)
    std_z = z(roll_std)
    diff_z = z(roll_diff)

    score = (1.2 * mean_z) - (0.8 * std_z) - (0.6 * diff_z)

    out = pd.DataFrame(
        {
            "timestamp": g["timestamp"].values,
            "window_end": g["timestamp"].values,
            "window_start": g["timestamp"].values,
            "score": score,
        }
    ).dropna(subset=["score"])

    # Pencere başlangıcını timestamp aralığından tahmin eder.
    if len(g) > 1:
        step = g["timestamp"].sort_values().diff().median()
        if pd.isna(step):
            step = pd.Timedelta(hours=1)
    else:
        step = pd.Timedelta(hours=1)

    out["window_start"] = out["window_end"] - step * (window_periods - 1)
    return out


def pick_recommendations(scada_agg: pd.DataFrame, window_periods: int, top_k: int, min_gap_hours: int) -> pd.DataFrame:
    recs = []
    for (montaj_key, tr_code_norm), g in scada_agg.groupby(["montaj_key", "tr_code_norm"]):
        s = score_windows(g, window_periods=window_periods).sort_values("score", ascending=False)

        chosen = []
        for _, row in s.iterrows():
            if len(chosen) >= top_k:
                break
            ok = True
            for c in chosen:
                if abs((row["window_start"] - c["window_start"]).total_seconds()) < min_gap_hours * 3600:
                    ok = False
                    break
            if ok:
                chosen.append(row)

        if chosen:
            r = pd.DataFrame(chosen)
            r.insert(0, "montaj_key", montaj_key)
            r.insert(1, "tr_code_norm", tr_code_norm)
            recs.append(r)

    if not recs:
        return pd.DataFrame(columns=["montaj_key", "tr_code_norm", "window_start", "window_end", "score"])

    out = pd.concat(recs, ignore_index=True)
    out = out.sort_values(["montaj_key", "tr_code_norm", "score"], ascending=[True, True, False])

    demand_at_end = scada_agg[["montaj_key", "tr_code_norm", "timestamp", "demand_kva"]].copy()
    out = out.merge(
        demand_at_end,
        left_on=["montaj_key", "tr_code_norm", "window_end"],
        right_on=["montaj_key", "tr_code_norm", "timestamp"],
        how="left",
    ).drop(columns=["timestamp"], errors="ignore")

    return out


# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="AYEDAŞ | Trafo Kapasite & Demand Paneli", layout="wide")

    st.title("⚡ AYEDAŞ Trafo Kapasite & Demand Karar Destek Paneli")
    st.caption("CBS kurulu güç verisi ile SCADA demand verisini eşleştirir, yüklenme oranı ve yeni bağlantı etkisini hesaplar.")

    scada_file = find_existing_file(SCADA_FILE_CANDIDATES)
    cbs_file = find_existing_file(CBS_FILE_CANDIDATES)

    with st.sidebar:
        st.header("Veri Kaynağı")
        st.caption("Excel dosyaları repo kök dizininde olmalı.")
        st.write("SCADA:", scada_file.name if scada_file else "Bulunamadı")
        st.write("CBS:", cbs_file.name if cbs_file else "Bulunamadı")

        st.divider()
        st.header("Temizleme")
        only_valid = st.toggle("Sadece 'Valid' kaliteyi kullan", value=True)
        remove_zeros = st.toggle("0 değerleri kaldır", value=True)
        only_cbs_scada = st.toggle("Sadece CBS'de SCADA-RTU = Evet olan trafolar", value=True)

        st.divider()
        st.header("Analiz Ayarları")
        freq_label = st.selectbox(
            "Zaman kovası",
            ["15 dakika", "Saatlik", "Günlük"],
            index=0,
            help="15 dakikalık canlı/veri exportu geldikçe 15 dakika seçili kalmalı. Eski günlük exportlarda Günlük daha okunaklı olabilir.",
        )
        freq = {"15 dakika": "15min", "Saatlik": "h", "Günlük": "D"}[freq_label]

        planning_metric = st.selectbox(
            "Planlama demand metriği",
            ["P95 demand", "Termal demand", "P99 demand", "Maks demand", "Ortalama demand"],
            index=0,
        )
        thermal_window_hours = st.slider("Termal pencere (saat)", 1, 24, 4)
        warn_pct = st.slider("Sınırda eşiği (%)", 50, 100, 80)
        critical_pct = st.slider("Kritik eşiği (%)", 70, 100, 95)

        st.divider()
        st.header("Yeni Bağlantı Simülasyonu")
        demand_kw = st.number_input("Yeni talep (kW)", min_value=1.0, value=50.0, step=10.0)
        power_factor = st.number_input("Güç faktörü", min_value=0.50, max_value=1.00, value=0.90, step=0.01)

        st.divider()
        st.header("Hamule Önerisi")
        window_hours = st.slider("Ölçüm penceresi (saat)", 1, 12, 2)
        top_k = st.slider("Trafo başına öneri sayısı", 1, 10, 3)
        min_gap = st.slider("Öneriler arası min boşluk (saat)", 1, 168, 24)

    if not scada_file:
        st.error(
            "SCADA Excel dosyası bulunamadı. Repo kök dizinine şu isimlerden biriyle yükleyin: "
            + ", ".join(SCADA_FILE_CANDIDATES)
        )
        st.stop()

    if not cbs_file:
        st.error(
            "CBS Excel dosyası bulunamadı. Repo kök dizinine şu isimlerden biriyle yükleyin: "
            + ", ".join(CBS_FILE_CANDIDATES)
        )
        st.stop()

    try:
        scada_raw = load_scada_excel(str(scada_file))
        cbs_raw = load_cbs_excel(str(cbs_file))
    except Exception as e:
        st.error(f"Dosyalar okunurken hata oluştu: {e}")
        st.stop()

    scada_clean = prepare_scada(scada_raw, only_valid=only_valid, remove_zeros=remove_zeros)
    scada_agg = aggregate_scada(scada_clean, freq=freq)

    period_hours = estimate_period_hours(freq)
    thermal_window_points = max(1, int(round(thermal_window_hours / period_hours)))
    hamule_window_periods = max(1, int(round(window_hours / period_hours)))

    analysis, cbs_no_scada_data, scada_no_cbs = build_capacity_analysis(
        cbs=cbs_raw,
        scada_agg=scada_agg,
        only_cbs_scada=only_cbs_scada,
        planning_metric=planning_metric,
        thermal_window_points=thermal_window_points,
        warn_pct=warn_pct,
        critical_pct=critical_pct,
    )

    sim = simulate_connection(analysis, demand_kw=demand_kw, power_factor=power_factor)

    matched = analysis[analysis["veri_adedi"].notna()].copy()
    risky = matched[matched["risk_durumu"].isin(["Sınırda", "Kritik", "Aşırı Yüklü"])]
    investment_needed = sim[sim["baglanti_karari"].eq("Yatırım Gerekli")]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("CBS SCADA kapsamı", f"{int(analysis['scada_rtu_var'].sum()):,}" if "scada_rtu_var" in analysis else "-")
    k2.metric("SCADA demand gelen", f"{scada_agg[['montaj_key', 'tr_code_norm']].drop_duplicates().shape[0]:,}")
    k3.metric("Eşleşen trafo", f"{len(matched):,}")
    k4.metric("Riskli trafo", f"{len(risky):,}")
    k5.metric("50 kW+ sim. yatırım", f"{len(investment_needed):,}")

    tab_capacity, tab_detail, tab_sim, tab_hamule, tab_quality = st.tabs(
        [
            "📊 Kapasite Analizi",
            "🔍 Trafo Detay",
            "🧮 Yeni Bağlantı Simülasyonu",
            "🎯 Hamule Ölçüm Önerisi",
            "🧹 Veri Kalitesi / Eşleşme",
        ]
    )

    with tab_capacity:
        st.subheader("Kapasite analizi özeti")
        st.info(
            "Bu tabloda CBS'de SCADA-RTU = Evet olan trafolar esas alınır. "
            "SCADA demand eşleşirse kurulu güç ile karşılaştırılır."
        )

        f1, f2, f3 = st.columns([1, 1, 2])
        with f1:
            risk_filter = st.multiselect(
                "Risk durumu",
                ["Normal", "Sınırda", "Kritik", "Aşırı Yüklü", "Veri Yok"],
                default=["Normal", "Sınırda", "Kritik", "Aşırı Yüklü", "Veri Yok"],
            )
        with f2:
            mahalleler = sorted([m for m in analysis["mahalle"].dropna().astype(str).unique() if m])
            mahalle_filter = st.multiselect("Mahalle", mahalleler)
        with f3:
            search = st.text_input("Montaj yeri / AssetID / Trafo kodu ara")

        view = analysis.copy()
        if risk_filter:
            view = view[view["risk_durumu"].isin(risk_filter)]
        if mahalle_filter:
            view = view[view["mahalle"].astype(str).isin(mahalle_filter)]
        if search.strip():
            s = search.strip().upper()
            mask = (
                view["montaj_yeri"].astype(str).str.upper().str.contains(s, na=False)
                | view["asset_id"].astype(str).str.upper().str.contains(s, na=False)
                | view["tr_code_norm"].astype(str).str.upper().str.contains(s, na=False)
            )
            view = view[mask]

        view = view.sort_values("yuklenme_pct", ascending=False, na_position="last")
        capacity_cols = [
            "montaj_yeri",
            "tr_code_norm",
            "asset_id",
            "mahalle",
            "kurulu_guc_kva",
            "planlama_demand_kva",
            "yuklenme_pct",
            "p95_demand_kva",
            "p99_demand_kva",
            "termal_demand_kva",
            "maks_demand_kva",
            "bos_kapasite_kva_80",
            "bos_kapasite_kva_100",
            "risk_durumu",
            "veri_adedi",
            "son_veri",
        ]
        st.dataframe(
            view[capacity_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "montaj_yeri": "Montaj Yeri",
                "tr_code_norm": "Trafo Kodu",
                "kurulu_guc_kva": st.column_config.NumberColumn("Kurulu Güç (kVA)", format="%.0f"),
                "planlama_demand_kva": st.column_config.NumberColumn("Planlama Demand (kVA)", format="%.2f"),
                "yuklenme_pct": st.column_config.ProgressColumn("Yüklenme %", min_value=0, max_value=120, format="%.1f%%"),
                "bos_kapasite_kva_80": st.column_config.NumberColumn("80% Eşiğe Boşluk (kVA)", format="%.2f"),
                "bos_kapasite_kva_100": st.column_config.NumberColumn("100% Eşiğe Boşluk (kVA)", format="%.2f"),
            },
        )

        csv = view[capacity_cols].to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Kapasite analizini indir", csv, "kapasite_analizi.csv", "text/csv")

        chart_data = view.dropna(subset=["yuklenme_pct"]).head(20).copy()
        if not chart_data.empty:
            chart_data["etiket"] = chart_data["montaj_yeri"].astype(str) + " / " + chart_data["tr_code_norm"].astype(str)
            fig = px.bar(
                chart_data.sort_values("yuklenme_pct", ascending=True),
                x="yuklenme_pct",
                y="etiket",
                orientation="h",
                hover_data=["kurulu_guc_kva", "planlama_demand_kva", "risk_durumu"],
                title="En yüksek yüklenme oranına sahip ilk 20 trafo",
            )
            fig.add_vline(x=warn_pct, line_dash="dash", annotation_text=f"{warn_pct}%")
            fig.add_vline(x=critical_pct, line_dash="dash", annotation_text=f"{critical_pct}%")
            fig.update_layout(yaxis_title="Trafo", xaxis_title="Yüklenme (%)")
            st.plotly_chart(fig, use_container_width=True)

    with tab_detail:
        st.subheader("Trafo detay analizi")
        selectable = matched.sort_values(["montaj_yeri", "tr_code_norm"]).copy()
        if selectable.empty:
            st.warning("Eşleşen trafo bulunamadı. Veri Kalitesi sekmesindeki eşleşmeyen kayıtları kontrol edin.")
        else:
            selectable["secim"] = selectable["montaj_yeri"].astype(str) + " / " + selectable["tr_code_norm"].astype(str)
            selected = st.selectbox("Trafo seç", selectable["secim"].tolist())
            row = selectable[selectable["secim"].eq(selected)].iloc[0]

            detail_ts = scada_agg[
                (scada_agg["montaj_key"].eq(row["montaj_key"]))
                & (scada_agg["tr_code_norm"].eq(row["tr_code_norm"]))
            ].sort_values("timestamp")

            d1, d2, d3, d4, d5 = st.columns(5)
            d1.metric("Kurulu güç", fmt_num(row["kurulu_guc_kva"], 0, " kVA"))
            d2.metric("P95 demand", fmt_num(row["p95_demand_kva"], 2, " kVA"))
            d3.metric("Termal demand", fmt_num(row["termal_demand_kva"], 2, " kVA"))
            d4.metric("Yüklenme", fmt_num(row["yuklenme_pct"], 1, "%"))
            d5.metric("Risk", row["risk_durumu"])

            st.write(
                f"**Mahalle:** {row.get('mahalle', '-')}  |  "
                f"**AssetID:** {row.get('asset_id', '-')}  |  "
                f"**SCADA noktası:** {row.get('dm_id', row['montaj_yeri'])} / {row.get('h_cell', '-')}"
            )

            if not detail_ts.empty:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=detail_ts["timestamp"],
                        y=detail_ts["demand_kva"],
                        mode="lines+markers",
                        name="Demand (kVA)",
                    )
                )
                fig.add_hline(y=row["kurulu_guc_kva"], line_dash="dash", annotation_text="100% kurulu güç")
                fig.add_hline(y=row["kurulu_guc_kva"] * warn_pct / 100, line_dash="dash", annotation_text=f"{warn_pct}% eşik")
                fig.update_layout(
                    title=f"{row['montaj_yeri']} / {row['tr_code_norm']} Demand Profili",
                    xaxis_title="Zaman",
                    yaxis_title="kVA",
                )
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(detail_ts, use_container_width=True, hide_index=True)

    with tab_sim:
        st.subheader("Yeni bağlantı talebi simülasyonu")
        st.write(
            f"Seçilen talep: **{demand_kw:.0f} kW**, güç faktörü: **{power_factor:.2f}**, "
            f"kVA karşılığı: **{demand_kw / power_factor:.2f} kVA**"
        )

        sim_view = sim.sort_values("projeksiyon_yuklenme_pct", ascending=False, na_position="last")
        sim_cols = [
            "montaj_yeri",
            "tr_code_norm",
            "mahalle",
            "kurulu_guc_kva",
            "planlama_demand_kva",
            "yuklenme_pct",
            "yeni_talep_kva",
            "projeksiyon_demand_kva",
            "projeksiyon_yuklenme_pct",
            "baglanti_karari",
        ]
        st.dataframe(
            sim_view[sim_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "yuklenme_pct": st.column_config.ProgressColumn("Mevcut Yüklenme %", min_value=0, max_value=120, format="%.1f%%"),
                "projeksiyon_yuklenme_pct": st.column_config.ProgressColumn("Projeksiyon %", min_value=0, max_value=120, format="%.1f%%"),
            },
        )
        csv_sim = sim_view[sim_cols].to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Bağlantı simülasyonunu indir", csv_sim, "baglanti_simulasyonu.csv", "text/csv")

        counts = sim_view["baglanti_karari"].value_counts().reset_index()
        counts.columns = ["baglanti_karari", "adet"]
        if not counts.empty:
            fig = px.pie(counts, names="baglanti_karari", values="adet", title="Yeni talep sonrası karar dağılımı")
            st.plotly_chart(fig, use_container_width=True)

    with tab_hamule:
        st.subheader("Hamule ölçümü için önerilen zaman pencereleri")
        recs = pick_recommendations(
            scada_agg,
            window_periods=hamule_window_periods,
            top_k=top_k,
            min_gap_hours=min_gap,
        )
        if recs.empty:
            st.warning("Öneri üretilemedi. Filtreleri gevşetmeyi veya zaman kovasını değiştirmeyi deneyin.")
        else:
            recs = recs.merge(
                cbs_raw[["montaj_key", "tr_code_norm", "montaj_yeri", "mahalle", "kurulu_guc_kva"]].drop_duplicates(
                    subset=["montaj_key", "tr_code_norm"]
                ),
                on=["montaj_key", "tr_code_norm"],
                how="left",
            )
            recs["score"] = recs["score"].round(3)
            if "demand_kva" in recs.columns:
                recs["demand_kva"] = recs["demand_kva"].round(2)
            rec_cols = ["montaj_yeri", "tr_code_norm", "mahalle", "window_start", "window_end", "demand_kva", "score"]
            st.dataframe(recs[rec_cols], use_container_width=True, hide_index=True)
            csv_rec = recs[rec_cols].to_csv(index=False).encode("utf-8-sig")
            st.download_button("⬇️ Hamule önerilerini indir", csv_rec, "hamule_olcum_onerileri.csv", "text/csv")

    with tab_quality:
        st.subheader("Veri kalitesi ve eşleşme kontrolü")

        q1, q2, q3, q4 = st.columns(4)
        q1.metric("SCADA ham kayıt", f"{len(scada_raw):,}")
        q2.metric("SCADA temiz kayıt", f"{len(scada_clean):,}")
        q3.metric("CBS toplam trafo", f"{len(cbs_raw):,}")
        q4.metric("CBS SCADA-RTU = Evet", f"{int(cbs_raw['scada_rtu_var'].sum()):,}")

        st.markdown("#### CBS'de SCADA-RTU = Evet ama SCADA demand verisi eşleşmeyenler")
        if cbs_no_scada_data.empty:
            st.success("Bu kategoride kayıt yok.")
        else:
            cols = ["montaj_yeri", "tr_code_norm", "asset_id", "mahalle", "kurulu_guc_kva", "SCADA-RTU Var mı?"]
            available_cols = [c for c in cols if c in cbs_no_scada_data.columns]
            st.dataframe(cbs_no_scada_data[available_cols], use_container_width=True, hide_index=True)

        st.markdown("#### SCADA demand var ama CBS karşılığı bulunamayanlar")
        if scada_no_cbs.empty:
            st.success("Bu kategoride kayıt yok.")
        else:
            st.dataframe(scada_no_cbs, use_container_width=True, hide_index=True)

        st.markdown("#### SCADA temiz veri örneği")
        st.dataframe(scada_clean.head(100), use_container_width=True, hide_index=True)

        with st.expander("🧠 Eşleşme mantığı"):
            st.write(
                "SCADA Point Name içindeki `T-4092` değeri CBS `Montaj Yeri = T4092` ile, "
                "`Enan1 / Enan2` ise CBS `Trafo Kodu = TR1 / TR2` ile eşleştirilir. "
                "`H01`, `H03`, `H04` gibi ifadeler trafo kodu değil, hücre/nokta detayı olarak saklanır."
            )


if __name__ == "__main__":
    main()
