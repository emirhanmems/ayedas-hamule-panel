import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="AYEDAŞ | Mahalle Bazlı Yük Aktarım Öneri Paneli",
    layout="wide",
)

# =========================================================
# DOSYA AYARLARI
# Streamlit repo içine bu dosyaları koyarsanız otomatik okunur.
# Dosya isimleri farklıysa panelde upload alanından da yükleyebilirsiniz.
# =========================================================
SCADA_FILE_CANDIDATES = [
    "Sancaktepe Trafo demand 2025.xlsx",
    "Sancaktepe Trafo demand 2025 (1).xlsx",
]

CBS_FILE_CANDIDATES = [
    "Trafo Sorgu Sonuçları.xlsx",
    "Trafo Sorgu Sonuçları (2).xlsx",
]


# =========================================================
# GENEL YARDIMCI FONKSİYONLAR
# =========================================================
def find_existing_file(candidates):
    for name in candidates:
        p = Path(name)
        if p.exists():
            return str(p)
    return None


def normalize_code(value):
    """T-4014, T4014, t 4014 gibi kodları T4014 formatına indirger."""
    if pd.isna(value):
        return np.nan
    s = str(value).upper().strip()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s if s else np.nan


def normalize_text(value):
    if pd.isna(value):
        return "BİLİNMİYOR"
    s = str(value).strip().upper()
    s = re.sub(r"\s+", " ", s)
    return s if s else "BİLİNMİYOR"


def yes_like(value):
    s = normalize_text(value)
    return s in {"EVET", "YES", "VAR", "1", "TRUE"}


def is_quality_valid(q):
    """Invalid içinde valid geçtiği için özellikle invalid dışlanır."""
    s = str(q).strip().lower()
    return ("valid" in s) and ("invalid" not in s)


def risk_level(load_pct):
    if pd.isna(load_pct):
        return "Demand Yok"
    if load_pct >= 100:
        return "Kritik"
    if load_pct >= 80:
        return "Yüksek"
    if load_pct >= 60:
        return "Orta"
    return "Düşük"


def risk_color(level):
    return {
        "Kritik": "Kırmızı",
        "Yüksek": "Turuncu",
        "Orta": "Sarı",
        "Düşük": "Yeşil",
        "Demand Yok": "Gri",
    }.get(level, "Gri")


def safe_num(series):
    return pd.to_numeric(series, errors="coerce")


# =========================================================
# SCADA DEMAND OKUMA
# =========================================================
@st.cache_data(show_spinner=False)
def load_scada_excel(file_obj_or_path):
    """
    SCADA export dosyasında header 1. satırda değilse otomatik bulur.
    Beklenen ana kolonlar: Point Name, Time stamp, Value, Source / Quality.
    """
    raw = pd.read_excel(file_obj_or_path, sheet_name=0, header=None, engine="openpyxl")

    header_row_idx = None
    for i in range(min(30, len(raw))):
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

    if "quality" not in df.columns:
        df["quality"] = "Unknown"

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["point_name"] = df["point_name"].astype(str)

    df = df.dropna(subset=["point_name", "timestamp", "value"])
    return df[["point_name", "timestamp", "value", "quality"]]


def parse_scada_point(point_name):
    """
    Örnek point:
    /Net-E/SANCAKTEPE OM DTM/T-4014/0.4kV/Enan1 H03/S

    location_code_raw = T-4014
    trafo_key         = T4014
    feeder_id         = H03
    metric            = S
    """
    p = str(point_name)

    location_code_raw = None
    m = re.search(r"/(T-?\d+)", p, flags=re.IGNORECASE)
    if m:
        location_code_raw = m.group(1).upper()

    feeder_id = None
    m2 = re.search(r"\b(H\d{2})\b", p, flags=re.IGNORECASE)
    if m2:
        feeder_id = m2.group(1).upper()

    metric = None
    m3 = re.search(r"/([A-Za-z0-9]+)\s*$", p.strip())
    if m3:
        metric = m3.group(1).upper()

    return location_code_raw, normalize_code(location_code_raw), feeder_id, metric


def prepare_scada_hourly(
    scada_raw,
    only_valid=True,
    remove_zeros=True,
    only_s_metric=True,
    feeder_aggregation="sum",
):
    d = scada_raw.copy()

    parsed = d["point_name"].apply(parse_scada_point)
    d["location_code_raw"] = parsed.apply(lambda x: x[0])
    d["trafo_key"] = parsed.apply(lambda x: x[1])
    d["feeder_id"] = parsed.apply(lambda x: x[2])
    d["metric"] = parsed.apply(lambda x: x[3])

    d = d.dropna(subset=["trafo_key"])

    if only_valid:
        d = d[d["quality"].apply(is_quality_valid)]

    if remove_zeros:
        d = d[d["value"] > 0]

    if only_s_metric:
        d = d[d["metric"].eq("S")]

    d["hour"] = d["timestamp"].dt.floor("h")

    # Aynı feeder aynı saat içinde birden çok kayıt varsa pik değeri al.
    feeder_hour = (
        d.groupby(["trafo_key", "location_code_raw", "feeder_id", "hour"], as_index=False)["value"]
        .max()
        .rename(columns={"value": "feeder_demand_kva"})
    )

    if feeder_aggregation == "sum":
        hourly = (
            feeder_hour.groupby(["trafo_key", "location_code_raw", "hour"], as_index=False)[
                "feeder_demand_kva"
            ]
            .sum()
            .rename(columns={"hour": "timestamp", "feeder_demand_kva": "demand_kva"})
        )
    else:
        hourly = (
            feeder_hour.groupby(["trafo_key", "location_code_raw", "hour"], as_index=False)[
                "feeder_demand_kva"
            ]
            .max()
            .rename(columns={"hour": "timestamp", "feeder_demand_kva": "demand_kva"})
        )

    hourly = hourly.sort_values(["trafo_key", "timestamp"])
    return d, feeder_hour, hourly


def summarize_demand(hourly):
    if hourly.empty:
        return pd.DataFrame(
            columns=[
                "trafo_key",
                "scada_code",
                "max_demand_kva",
                "peak_time",
                "avg_demand_kva",
                "latest_demand_kva",
                "latest_time",
                "data_points",
            ]
        )

    h = hourly.copy().sort_values(["trafo_key", "timestamp"])

    peak_idx = h.groupby("trafo_key")["demand_kva"].idxmax()
    peak = h.loc[peak_idx, ["trafo_key", "location_code_raw", "timestamp", "demand_kva"]].rename(
        columns={
            "location_code_raw": "scada_code",
            "timestamp": "peak_time",
            "demand_kva": "max_demand_kva",
        }
    )

    latest = h.groupby("trafo_key", as_index=False).tail(1)[
        ["trafo_key", "timestamp", "demand_kva"]
    ].rename(columns={"timestamp": "latest_time", "demand_kva": "latest_demand_kva"})

    stats = (
        h.groupby("trafo_key", as_index=False)
        .agg(avg_demand_kva=("demand_kva", "mean"), data_points=("demand_kva", "size"))
    )

    out = peak.merge(latest, on="trafo_key", how="left").merge(stats, on="trafo_key", how="left")
    return out


# =========================================================
# CBS TRAFO VERİSİ OKUMA
# =========================================================
@st.cache_data(show_spinner=False)
def load_cbs_excel(file_obj_or_path):
    df = pd.read_excel(file_obj_or_path, sheet_name=0, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_col(df, candidates, required=True):
    norm_cols = {normalize_text(c): c for c in df.columns}
    for cand in candidates:
        key = normalize_text(cand)
        if key in norm_cols:
            return norm_cols[key]

    # Daha esnek arama
    for c in df.columns:
        nc = normalize_text(c)
        for cand in candidates:
            if normalize_text(cand) in nc:
                return c

    if required:
        raise ValueError(f"CBS dosyasında beklenen kolon bulunamadı. Adaylar: {candidates}")
    return None


def prepare_cbs_site(cbs_raw):
    """
    CBS'de aynı Montaj Yeri altında TR1/TR2 olabilir.
    SCADA T-xxxx seviyesiyle eşleşmek için Montaj Yeri bazında toplanmış kapasite kullanılır.
    """
    df = cbs_raw.copy()

    col_location = find_col(df, ["Montaj Yeri"])
    col_capacity = find_col(df, ["Gücü[kVA]", "Gücü", "Guc", "Güç"])
    col_neighborhood = find_col(df, ["Mahalle"])
    col_district = find_col(df, ["İlçe", "Ilce"], required=False)
    col_asset = find_col(df, ["AssetID", "Asset ID"], required=False)
    col_trafo_code = find_col(df, ["Trafo Kodu"], required=False)
    col_osos = find_col(df, ["Trafo-OSOS Var mı?", "OSOS"], required=False)
    col_scada = find_col(df, ["SCADA-RTU Var mı?", "SCADA"], required=False)

    c = pd.DataFrame()
    c["trafo_key"] = df[col_location].apply(normalize_code)
    c["montaj_yeri"] = df[col_location].astype(str).str.strip()
    c["capacity_kva"] = safe_num(df[col_capacity])
    c["mahalle"] = df[col_neighborhood].apply(normalize_text)

    if col_district:
        c["ilce"] = df[col_district].apply(normalize_text)
    else:
        c["ilce"] = "BİLİNMİYOR"

    if col_asset:
        asset_series = df[col_asset]
        c["asset_id"] = asset_series.astype(str)
        # AssetID boşsa bütün boşları tek kayıt sanmasın diye satır bazlı benzersiz id ver.
        c.loc[asset_series.isna(), "asset_id"] = [
            f"row_{i}" for i in c.index[asset_series.isna()]
        ]
    else:
        c["asset_id"] = [f"row_{i}" for i in range(len(df))]

    if col_trafo_code:
        c["trafo_kodu"] = df[col_trafo_code].astype(str)
    else:
        c["trafo_kodu"] = ""

    if col_osos:
        c["osos_var"] = df[col_osos].apply(yes_like)
    else:
        c["osos_var"] = False

    if col_scada:
        c["cbs_scada_var"] = df[col_scada].apply(yes_like)
    else:
        c["cbs_scada_var"] = False

    c = c.dropna(subset=["trafo_key", "capacity_kva"])
    c = c[c["capacity_kva"] > 0]

    # Aynı AssetID birebir tekrar geldiyse çift saymasın.
    c = c.drop_duplicates(subset=["asset_id"])

    def join_unique(values):
        vals = sorted({str(v) for v in values if str(v).strip() and str(v).lower() != "nan"})
        return ", ".join(vals)

    site = (
        c.groupby("trafo_key", as_index=False)
        .agg(
            montaj_yeri=("montaj_yeri", "first"),
            mahalle=("mahalle", "first"),
            ilce=("ilce", "first"),
            capacity_kva=("capacity_kva", "sum"),
            trafo_adedi=("asset_id", "nunique"),
            trafo_kodlari=("trafo_kodu", join_unique),
            osos_var=("osos_var", "max"),
            cbs_scada_var=("cbs_scada_var", "max"),
        )
        .sort_values("trafo_key")
    )

    return site


# =========================================================
# RİSK VE YÜK AKTARIM ÖNERİSİ
# =========================================================
def build_risk_table(cbs_site, demand_summary):
    panel = cbs_site.merge(demand_summary, on="trafo_key", how="left")

    panel["load_pct"] = panel["max_demand_kva"] / panel["capacity_kva"] * 100
    panel["risk_level"] = panel["load_pct"].apply(risk_level)
    panel["risk_color"] = panel["risk_level"].apply(risk_color)
    panel["has_scada_demand"] = panel["max_demand_kva"].notna()

    panel["avg_demand_kva"] = panel["avg_demand_kva"].round(2)
    panel["max_demand_kva"] = panel["max_demand_kva"].round(2)
    panel["latest_demand_kva"] = panel["latest_demand_kva"].round(2)
    panel["load_pct"] = panel["load_pct"].round(2)

    return panel


def make_transfer_recommendations(
    risk_table,
    source_min_pct=100,
    source_target_pct=90,
    candidate_max_pct=80,
    top_n=5,
):
    """
    Kritik/yüksek trafo için aynı mahalledeki boş kapasiteli trafoları önerir.
    Bu gerçek AG topolojisi yerine mahalle bazlı aday önerisidir.
    """
    df = risk_table.copy()
    df = df[df["has_scada_demand"]].copy()
    df = df.dropna(subset=["mahalle", "capacity_kva", "max_demand_kva", "load_pct"])

    recs = []
    sources = df[df["load_pct"] >= source_min_pct].copy()

    for _, src in sources.iterrows():
        source_target_kva = src["capacity_kva"] * source_target_pct / 100
        source_excess_kva = max(float(src["max_demand_kva"] - source_target_kva), 0.0)

        if source_excess_kva <= 0:
            continue

        candidates = df[
            (df["mahalle"] == src["mahalle"])
            & (df["trafo_key"] != src["trafo_key"])
            & (df["load_pct"] < candidate_max_pct)
        ].copy()

        if candidates.empty:
            continue

        candidates["candidate_allowed_kva"] = candidates["capacity_kva"] * candidate_max_pct / 100
        candidates["spare_kva"] = candidates["candidate_allowed_kva"] - candidates["max_demand_kva"]
        candidates = candidates[candidates["spare_kva"] > 0].copy()

        if candidates.empty:
            continue

        for _, cand in candidates.iterrows():
            transfer_kva = min(source_excess_kva, float(cand["spare_kva"]))
            source_after_kva = float(src["max_demand_kva"] - transfer_kva)
            candidate_after_kva = float(cand["max_demand_kva"] + transfer_kva)
            source_after_pct = source_after_kva / float(src["capacity_kva"]) * 100
            candidate_after_pct = candidate_after_kva / float(cand["capacity_kva"]) * 100

            # 0-100 arası basit uygunluk skoru
            excess_cover_score = min(transfer_kva / max(source_excess_kva, 1), 1) * 45
            low_load_score = max(0, (candidate_max_pct - cand["load_pct"]) / candidate_max_pct) * 25
            spare_score = min(cand["spare_kva"] / max(source_excess_kva, 1), 1) * 20
            data_score = (5 if bool(cand.get("osos_var", False)) else 0) + (
                5 if bool(cand.get("cbs_scada_var", False)) else 0
            )
            suitability_score = excess_cover_score + low_load_score + spare_score + data_score

            recs.append(
                {
                    "source_trafo": src["montaj_yeri"],
                    "source_key": src["trafo_key"],
                    "mahalle": src["mahalle"],
                    "source_capacity_kva": src["capacity_kva"],
                    "source_max_demand_kva": src["max_demand_kva"],
                    "source_load_pct": src["load_pct"],
                    "source_target_pct": source_target_pct,
                    "source_excess_kva": source_excess_kva,
                    "candidate_trafo": cand["montaj_yeri"],
                    "candidate_key": cand["trafo_key"],
                    "candidate_capacity_kva": cand["capacity_kva"],
                    "candidate_max_demand_kva": cand["max_demand_kva"],
                    "candidate_load_pct": cand["load_pct"],
                    "candidate_spare_kva": cand["spare_kva"],
                    "transferable_kva": transfer_kva,
                    "source_after_pct": source_after_pct,
                    "candidate_after_pct": candidate_after_pct,
                    "suitability_score": suitability_score,
                    "candidate_scada_var": cand.get("cbs_scada_var", False),
                    "candidate_osos_var": cand.get("osos_var", False),
                }
            )

    if not recs:
        return pd.DataFrame()

    out = pd.DataFrame(recs)
    out = out.sort_values(
        ["source_load_pct", "source_key", "suitability_score"],
        ascending=[False, True, False],
    )
    out["rank"] = out.groupby("source_key").cumcount() + 1
    out = out[out["rank"] <= top_n].copy()

    numeric_cols = [
        "source_capacity_kva",
        "source_max_demand_kva",
        "source_load_pct",
        "source_excess_kva",
        "candidate_capacity_kva",
        "candidate_max_demand_kva",
        "candidate_load_pct",
        "candidate_spare_kva",
        "transferable_kva",
        "source_after_pct",
        "candidate_after_pct",
        "suitability_score",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(2)

    return out


def add_best_recommendation_to_risk_table(risk_table, recs):
    table = risk_table.copy()
    if recs.empty:
        table["best_candidate"] = "Aday yok"
        table["best_transferable_kva"] = np.nan
        table["best_candidate_after_pct"] = np.nan
        table["onerilen_aksiyon"] = np.where(
            table["load_pct"] >= 100,
            "Kritik yüklü. Aynı mahallede uygun aday bulunamadı; saha/topoloji kontrolü ve güç artırımı değerlendirilmeli.",
            np.where(
                table["load_pct"] >= 80,
                "Yüksek yüklü. İzleme ve saha kontrolü önerilir.",
                "Normal izleme.",
            ),
        )
        return table

    best = recs.sort_values("suitability_score", ascending=False).groupby("source_key", as_index=False).first()
    best = best[
        [
            "source_key",
            "candidate_trafo",
            "transferable_kva",
            "candidate_after_pct",
            "source_after_pct",
            "suitability_score",
        ]
    ].rename(
        columns={
            "source_key": "trafo_key",
            "candidate_trafo": "best_candidate",
            "transferable_kva": "best_transferable_kva",
            "candidate_after_pct": "best_candidate_after_pct",
            "source_after_pct": "best_source_after_pct",
            "suitability_score": "best_suitability_score",
        }
    )

    table = table.merge(best, on="trafo_key", how="left")
    table["best_candidate"] = table["best_candidate"].fillna("Aday yok")

    def action(row):
        if pd.isna(row["load_pct"]):
            return "Demand verisi yok; SCADA/OSOS veri kontrolü gerekli."
        if row["load_pct"] >= 100:
            if row["best_candidate"] != "Aday yok":
                return (
                    f"Kritik yüklü. Aynı mahallede {row['best_candidate']} adaydır. "
                    f"Yaklaşık {row['best_transferable_kva']:.0f} kVA aktarım saha/topoloji kontrolüyle incelenmeli."
                )
            return "Kritik yüklü. Uygun aday yok; güç artırımı veya farklı mahalle/hat topolojisi incelenmeli."
        if row["load_pct"] >= 80:
            return "Yüksek yüklü. Saha kontrolü ve mahalle içi adaylar takip edilmeli."
        if row["load_pct"] >= 60:
            return "Orta risk. Periyodik izleme önerilir."
        return "Düşük risk. Normal izleme."

    table["onerilen_aksiyon"] = table.apply(action, axis=1)
    return table


# =========================================================
# UI BAŞLANGICI
# =========================================================
st.title("⚡ CBS Entegre Mahalle Bazlı Trafo Risk ve Yük Aktarım Öneri Paneli")
st.caption(
    "Bu panel, SCADA demand verisi ile CBS trafo bilgilerini eşleştirir; "
    "riskli trafoları bulur ve aynı mahallede boş kapasitesi olan trafoları yük aktarım adayı olarak önerir."
)

with st.sidebar:
    st.header("1) Veri Kaynağı")

    auto_scada = find_existing_file(SCADA_FILE_CANDIDATES)
    auto_cbs = find_existing_file(CBS_FILE_CANDIDATES)

    scada_upload = st.file_uploader("SCADA demand Excel", type=["xlsx", "xls"])
    cbs_upload = st.file_uploader("CBS trafo sorgu Excel", type=["xlsx", "xls"])

    st.caption(
        "Repo içinde dosya varsa otomatik okunur. Upload yaparsanız upload edilen dosya kullanılır."
    )

    st.divider()
    st.header("2) Temizleme")
    only_valid = st.toggle("Sadece Valid kalite", value=True)
    remove_zeros = st.toggle("0 demand değerlerini kaldır", value=True)
    only_s_metric = st.toggle("Sadece S/kVA metriklerini kullan", value=True)
    feeder_mode_label = st.radio(
        "Aynı T kodunda birden fazla H varsa",
        ["H hücrelerini topla", "H hücrelerinden en büyüğünü al"],
        index=0,
    )
    feeder_aggregation = "sum" if feeder_mode_label.startswith("H hücrelerini topla") else "max"

    st.divider()
    st.header("3) Risk / Öneri Ayarları")
    source_min_pct = st.slider("Öneri üretilecek min. trafo yüklenmesi (%)", 60, 150, 100, 5)
    source_target_pct = st.slider("Riskli trafo hedef yüklenmesi (%)", 60, 100, 90, 5)
    candidate_max_pct = st.slider("Aday trafo üst yük sınırı (%)", 50, 95, 80, 5)
    top_n = st.slider("Riskli trafo başına aday sayısı", 1, 10, 5)


scada_source = scada_upload if scada_upload is not None else auto_scada
cbs_source = cbs_upload if cbs_upload is not None else auto_cbs

if scada_source is None or cbs_source is None:
    st.error(
        "SCADA ve CBS Excel dosyaları bulunamadı. Dosyaları Streamlit repo içine koyun veya soldan upload edin."
    )
    st.info(
        "Beklenen örnek dosya adları: "
        f"SCADA: {SCADA_FILE_CANDIDATES[0]} | CBS: {CBS_FILE_CANDIDATES[0]}"
    )
    st.stop()

try:
    scada_raw = load_scada_excel(scada_source)
    cbs_raw = load_cbs_excel(cbs_source)
except Exception as e:
    st.error(f"Dosyalar okunamadı: {e}")
    st.stop()

try:
    scada_clean, feeder_hour, hourly = prepare_scada_hourly(
        scada_raw,
        only_valid=only_valid,
        remove_zeros=remove_zeros,
        only_s_metric=only_s_metric,
        feeder_aggregation=feeder_aggregation,
    )
    demand_summary = summarize_demand(hourly)
    cbs_site = prepare_cbs_site(cbs_raw)
    risk_table = build_risk_table(cbs_site, demand_summary)
    recommendations = make_transfer_recommendations(
        risk_table,
        source_min_pct=source_min_pct,
        source_target_pct=source_target_pct,
        candidate_max_pct=candidate_max_pct,
        top_n=top_n,
    )
    export_table = add_best_recommendation_to_risk_table(risk_table, recommendations)
except Exception as e:
    st.error(f"Veri işleme sırasında hata oluştu: {e}")
    st.stop()

matched_count = int(risk_table["has_scada_demand"].sum())
critical_count = int((risk_table["risk_level"] == "Kritik").sum())
high_count = int((risk_table["risk_level"] == "Yüksek").sum())
rec_source_count = 0 if recommendations.empty else int(recommendations["source_key"].nunique())

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("CBS trafo merkezi", f"{risk_table['trafo_key'].nunique():,}")
k2.metric("SCADA eşleşen", f"{matched_count:,}")
k3.metric("Kritik trafo", f"{critical_count:,}")
k4.metric("Yüksek risk", f"{high_count:,}")
k5.metric("Aday önerisi çıkan", f"{rec_source_count:,}")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Risk Özeti",
        "🔁 Mahalle Bazlı Yük Aktarımı",
        "🗺️ CBS Export Tablosu",
        "🔎 Veri Kontrol",
    ]
)

with tab1:
    st.subheader("Riskli Trafolar")

    c1, c2, c3 = st.columns(3)
    mahalle_options = ["TÜMÜ"] + sorted(risk_table["mahalle"].dropna().unique().tolist())
    selected_mahalle_overview = c1.selectbox("Mahalle filtresi", mahalle_options, key="overview_mahalle")
    risk_options = ["TÜMÜ", "Kritik", "Yüksek", "Orta", "Düşük", "Demand Yok"]
    selected_risk = c2.selectbox("Risk filtresi", risk_options)
    only_matched = c3.toggle("Sadece SCADA demand eşleşenleri göster", value=True)

    view = export_table.copy()
    if selected_mahalle_overview != "TÜMÜ":
        view = view[view["mahalle"] == selected_mahalle_overview]
    if selected_risk != "TÜMÜ":
        view = view[view["risk_level"] == selected_risk]
    if only_matched:
        view = view[view["has_scada_demand"]]

    view = view.sort_values("load_pct", ascending=False, na_position="last")

    show_cols = [
        "montaj_yeri",
        "mahalle",
        "capacity_kva",
        "max_demand_kva",
        "peak_time",
        "load_pct",
        "risk_level",
        "best_candidate",
        "best_transferable_kva",
        "onerilen_aksiyon",
    ]
    st.dataframe(view[show_cols], use_container_width=True, hide_index=True)

    st.subheader("Mahalle Bazlı Risk Yoğunluğu")
    risk_by_mahalle = (
        risk_table[risk_table["has_scada_demand"]]
        .groupby(["mahalle", "risk_level"], as_index=False)
        .agg(trafo_sayisi=("trafo_key", "nunique"), ort_yuklenme=("load_pct", "mean"))
    )

    if not risk_by_mahalle.empty:
        fig = px.bar(
            risk_by_mahalle,
            x="mahalle",
            y="trafo_sayisi",
            color="risk_level",
            barmode="stack",
            title="Mahalleye göre trafo risk dağılımı",
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Aynı Mahallede Yük Aktarım Adayı Önerisi")
    st.warning(
        "Bu öneri elektriksel tek hat/topoloji yerine mahalle + kapasite + demand bilgisiyle üretilir. "
        "Gerçek aktarım öncesi AG bağlantısı, kablo kesiti, faz dengesi, gerilim düşümü ve saha uygunluğu kontrol edilmelidir."
    )

    if recommendations.empty:
        st.info("Mevcut ayarlara göre uygun yük aktarım adayı bulunamadı. Eşik değerlerini gevşetmeyi deneyin.")
    else:
        risky_sources = (
            recommendations[["source_trafo", "source_key", "mahalle", "source_load_pct"]]
            .drop_duplicates()
            .sort_values("source_load_pct", ascending=False)
        )

        col_a, col_b = st.columns(2)
        mahalle_transfer_options = ["TÜMÜ"] + sorted(risky_sources["mahalle"].unique().tolist())
        selected_mahalle_transfer = col_a.selectbox(
            "Mahalle seç", mahalle_transfer_options, key="transfer_mahalle"
        )

        source_view = risky_sources.copy()
        if selected_mahalle_transfer != "TÜMÜ":
            source_view = source_view[source_view["mahalle"] == selected_mahalle_transfer]

        source_labels = (
            source_view["source_trafo"]
            + " | "
            + source_view["mahalle"]
            + " | %"
            + source_view["source_load_pct"].round(1).astype(str)
        ).tolist()

        if not source_labels:
            st.info("Bu mahallede öneri çıkan riskli trafo yok.")
        else:
            selected_label = col_b.selectbox("Riskli trafo seç", source_labels)
            selected_source_key = source_view.iloc[source_labels.index(selected_label)]["source_key"]

            rec_sel = recommendations[recommendations["source_key"] == selected_source_key].copy()
            rec_sel = rec_sel.sort_values("suitability_score", ascending=False)

            source_row = risk_table[risk_table["trafo_key"] == selected_source_key].iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Riskli trafo", source_row["montaj_yeri"])
            m2.metric("Mahalle", source_row["mahalle"])
            m3.metric("Mevcut yüklenme", f"%{source_row['load_pct']:.1f}")
            m4.metric("Maks demand", f"{source_row['max_demand_kva']:.1f} kVA")

            st.markdown("#### Önerilen aday trafolar")
            table_cols = [
                "rank",
                "candidate_trafo",
                "candidate_capacity_kva",
                "candidate_max_demand_kva",
                "candidate_load_pct",
                "candidate_spare_kva",
                "transferable_kva",
                "source_after_pct",
                "candidate_after_pct",
                "suitability_score",
            ]
            st.dataframe(rec_sel[table_cols], use_container_width=True, hide_index=True)

            fig2 = px.bar(
                rec_sel,
                x="candidate_trafo",
                y="transferable_kva",
                color="candidate_load_pct",
                hover_data=["candidate_spare_kva", "candidate_after_pct", "suitability_score"],
                title="Aday trafolara aktarılabilecek yaklaşık yük miktarı",
            )
            st.plotly_chart(fig2, use_container_width=True)

            best = rec_sel.iloc[0]
            st.success(
                f"En güçlü aday: {best['candidate_trafo']} | "
                f"Yaklaşık aktarılabilir yük: {best['transferable_kva']:.0f} kVA | "
                f"Aday trafo aktarım sonrası: %{best['candidate_after_pct']:.1f}"
            )

        st.markdown("#### Tüm öneri listesi")
        st.dataframe(recommendations, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("CBS'ye Aktarılabilecek Çıktı Tablosu")
    st.caption(
        "Bu tablo CBS tarafına join edilecek ana alanları içerir. Ana eşleşme alanı: trafo_key / Montaj Yeri normalize kodu."
    )

    cbs_export_cols = [
        "trafo_key",
        "montaj_yeri",
        "mahalle",
        "ilce",
        "capacity_kva",
        "trafo_adedi",
        "trafo_kodlari",
        "max_demand_kva",
        "peak_time",
        "load_pct",
        "risk_level",
        "risk_color",
        "best_candidate",
        "best_transferable_kva",
        "best_source_after_pct",
        "best_candidate_after_pct",
        "onerilen_aksiyon",
        "osos_var",
        "cbs_scada_var",
        "has_scada_demand",
    ]

    existing_cols = [c for c in cbs_export_cols if c in export_table.columns]
    export_view = export_table[existing_cols].sort_values("load_pct", ascending=False, na_position="last")
    st.dataframe(export_view, use_container_width=True, hide_index=True)

    csv = export_view.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ CBS çıktı tablosunu indir (CSV)",
        data=csv,
        file_name="cbs_mahalle_bazli_yuk_aktarim_oneri.csv",
        mime="text/csv",
    )

    if not recommendations.empty:
        rec_csv = recommendations.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Detaylı aday önerilerini indir (CSV)",
            data=rec_csv,
            file_name="detayli_yuk_aktarim_adaylari.csv",
            mime="text/csv",
        )

with tab4:
    st.subheader("Veri Eşleşme ve Kalite Kontrol")

    scada_keys = set(demand_summary["trafo_key"].dropna().unique())
    cbs_keys = set(cbs_site["trafo_key"].dropna().unique())
    unmatched_scada = sorted(scada_keys - cbs_keys)
    no_demand_cbs = sorted(cbs_keys - scada_keys)

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("SCADA ham kayıt", f"{len(scada_raw):,}")
    q2.metric("SCADA temiz kayıt", f"{len(scada_clean):,}")
    q3.metric("SCADA-CBS eşleşmeyen", f"{len(unmatched_scada):,}")
    q4.metric("CBS'de olup demand yok", f"{len(no_demand_cbs):,}")

    with st.expander("SCADA'da olup CBS'de eşleşmeyen T kodları"):
        st.write(unmatched_scada)

    with st.expander("CBS'de olup SCADA demand bulunmayan ilk 200 T kodu"):
        st.write(no_demand_cbs[:200])

    with st.expander("Temizlenmiş SCADA örneği"):
        st.dataframe(scada_clean.head(100), use_container_width=True, hide_index=True)

    with st.expander("CBS site/trafo merkezi örneği"):
        st.dataframe(cbs_site.head(100), use_container_width=True, hide_index=True)

    with st.expander("Yöntem notu"):
        st.markdown(
            """
            **Risk hesabı:**  
            `Yüklenme (%) = Maksimum Demand / Toplam Trafo Gücü × 100`

            **Yük aktarım önerisi:**  
            Sistem, riskli trafoyla aynı mahallede bulunan ve seçilen aday üst yük sınırının altında kalan trafoları tarar.  
            Adayın boş kapasitesi hesaplanır ve riskli trafodan ne kadar yük aktarılabileceği yaklaşık olarak önerilir.

            **Sınır:**  
            Bu çıktı kesin bağlantı kararı değildir. Tek hat/AG topoloji, kablo kesiti, faz dengesi ve gerilim düşümü kontrol edilmeden uygulanmamalıdır.
            """
        )
