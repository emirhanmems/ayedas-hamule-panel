import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_VERSION = "v5-risk-transfer-kva"

SCADA_FILE_CANDIDATES = [
    "Sancaktepe Trafo demand 2025.xlsx",
    "Sancaktepe Trafo demand 2025 yeni.xlsx",
    "Sancaktepe Trafo demand 2025 yeni(1).xlsx",
]

CBS_FILE_CANDIDATES = [
    "Trafo Sorgu Sonuçları.xlsx",
    "Trafo Sorgu Sonuclari.xlsx",
    "Trafo Sorgu Sonuçları(1).xlsx",
    "Trafo Sorgu Sonuçları(2).xlsx",
]

LOW_RISK_LIMIT = 60
MEDIUM_RISK_LIMIT = 80
HIGH_RISK_LIMIT = 100

DATA_ERROR_ABS_MAX_KVA = 10000
DATA_ERROR_CAP_MULTIPLIER = 2.5


def txt(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def norm_col(x) -> str:
    s = txt(x).lower()
    tr_map = str.maketrans("çğıöşüıİ", "cgiosuii")
    s = s.translate(tr_map)
    return re.sub(r"[^a-z0-9]+", "", s)


def norm_key(x) -> str:
    return re.sub(r"[^A-Z0-9]", "", txt(x).upper())


def norm_tr(x) -> str:
    s = norm_key(x)
    m = re.search(r"TR0*(\d+)", s)
    if m:
        return f"TR{int(m.group(1))}"
    return s


def yes(x) -> bool:
    return txt(x).lower() in {"evet", "e", "yes", "y", "true", "1", "var"}


def valid_quality(x) -> bool:
    s = txt(x).lower()
    return ("valid" in s) and ("invalid" not in s)


def find_file(candidates):
    for name in candidates:
        p = Path(name)
        if p.exists():
            return p
    return None


def find_col(columns, alternatives):
    normalized = {norm_col(c): c for c in columns}

    for alt in alternatives:
        key = norm_col(alt)
        if key in normalized:
            return normalized[key]

    for c in columns:
        c_norm = norm_col(c)
        for alt in alternatives:
            a_norm = norm_col(alt)
            if a_norm and a_norm in c_norm:
                return c

    return None


def fmt(x, digits=1, suffix=""):
    if pd.isna(x):
        return "-"
    try:
        return f"{float(x):,.{digits}f}{suffix}"
    except Exception:
        return "-"


def risk_level(load_pct):
    if pd.isna(load_pct):
        return "SCADA Verisi Yok"
    if load_pct > HIGH_RISK_LIMIT:
        return "Kritik Risk"
    if load_pct >= MEDIUM_RISK_LIMIT:
        return "Yüksek Risk"
    if load_pct >= LOW_RISK_LIMIT:
        return "Orta Risk"
    return "Düşük Risk"


def risk_order_value(risk):
    order = {
        "Kritik Risk": 4,
        "Yüksek Risk": 3,
        "Orta Risk": 2,
        "Düşük Risk": 1,
        "SCADA Verisi Yok": 0,
    }
    return order.get(txt(risk), 0)


def connection_decision(projected_pct):
    if pd.isna(projected_pct):
        return "SCADA Verisi Yok"
    if projected_pct > 100:
        return "Yatırım Gerekli"
    if projected_pct >= 80:
        return "Şartlı Uygun / İzlenmeli"
    return "Uygun"


@st.cache_data(show_spinner=False)
def load_scada_excel(path: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=0, header=None)

    header_row = None
    for i in range(min(30, len(raw))):
        row_text = " | ".join(txt(v).lower() for v in raw.iloc[i].tolist())
        if "point name" in row_text and "time stamp" in row_text:
            header_row = i
            break

    if header_row is None:
        raise ValueError("SCADA dosyasında 'Point Name' ve 'Time stamp' başlık satırı bulunamadı.")

    df = raw.iloc[header_row + 1:].copy()
    df.columns = [txt(c) for c in raw.iloc[header_row].tolist()]

    point_col = find_col(df.columns, ["Point Name"])
    time_col = find_col(df.columns, ["Time stamp", "Timestamp", "Tarih"])
    value_col = find_col(df.columns, ["Value", "Değer", "Deger"])
    quality_col = find_col(df.columns, ["Source / Quality", "Quality", "Kalite"])

    missing = []
    if point_col is None:
        missing.append("Point Name")
    if time_col is None:
        missing.append("Time stamp")
    if value_col is None:
        missing.append("Value")

    if missing:
        raise ValueError(
            f"SCADA dosyasında beklenen kolonlar yok: {missing}. Mevcut kolonlar: {list(df.columns)}"
        )

    out = pd.DataFrame(
        {
            "point_name": df[point_col].map(txt),
            "timestamp": pd.to_datetime(df[time_col], errors="coerce"),
            "value": pd.to_numeric(df[value_col], errors="coerce"),
            "quality": df[quality_col].map(txt) if quality_col else "Unknown",
        }
    )

    out = out.dropna(subset=["timestamp", "value"])
    out = out[out["point_name"].ne("")]
    return out


def parse_scada_point(point_name: str):
    p = txt(point_name)
    parts = p.split("/")

    station_part = ""
    for i, part in enumerate(parts):
        if "0.4" in txt(part).lower() and i > 0:
            station_part = txt(parts[i - 1])
            break

    if not station_part:
        station_part = p

    montaj_key = ""
    dm_label = ""

    m = re.search(r"\bT[-_ ]?(\d{3,6})\b", station_part, flags=re.I)
    if m:
        montaj_key = "T" + m.group(1)
        dm_label = "T-" + m.group(1)
    else:
        m = re.search(r"\bRP[-_ ]?(\d{3,6})\b", station_part, flags=re.I)
        if m:
            montaj_key = "B" + m.group(1)
            dm_label = "RP_" + m.group(1)
        else:
            m = re.search(r"\b(\d{3,6})\s*DM\b", station_part, flags=re.I)
            if m:
                montaj_key = "B" + m.group(1)
                dm_label = m.group(1) + " DM"
            else:
                montaj_key = norm_key(station_part)
                dm_label = station_part

    m_enan = re.search(r"\bEnan\s*[-_ ]?(\d+)\b|\bEnan(\d+)\b", p, flags=re.I)
    enan_no = ""
    if m_enan:
        enan_no = next((g for g in m_enan.groups() if g), "")

    tr_code = f"TR{int(enan_no)}" if enan_no else ""

    m_h = re.search(r"\bH\s*[-_ ]?(\d{1,3})\b", p, flags=re.I)
    h_cell = f"H{int(m_h.group(1)):02d}" if m_h else ""

    metric = ""
    m_metric = re.search(r"/([A-Za-z0-9]+)\s*$", p)
    if m_metric:
        metric = m_metric.group(1).upper()

    return montaj_key, dm_label, tr_code, h_cell, metric


def prepare_scada(raw: pd.DataFrame, only_valid=True, remove_zeros=True) -> pd.DataFrame:
    d = raw.copy()

    parsed = d["point_name"].apply(parse_scada_point)
    d["montaj_key"] = parsed.apply(lambda x: x[0])
    d["dm_label"] = parsed.apply(lambda x: x[1])
    d["tr_code"] = parsed.apply(lambda x: x[2])
    d["h_cell"] = parsed.apply(lambda x: x[3])
    d["metric"] = parsed.apply(lambda x: x[4])

    d = d[d["montaj_key"].ne("") & d["tr_code"].ne("")]
    d = d[d["metric"].eq("S") | d["metric"].eq("")]

    if only_valid:
        d = d[d["quality"].apply(valid_quality)]

    if remove_zeros:
        d = d[d["value"] > 0]

    return d


@st.cache_data(show_spinner=False)
def load_cbs_excel(path: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=0)
    raw.columns = [txt(c) for c in raw.columns]

    montaj_col = find_col(raw.columns, ["Montaj Yeri"])
    guc_col = find_col(raw.columns, ["Gücü[kVA]", "Gucu[kVA]", "Gücü", "Gucu"])
    tr_col = find_col(raw.columns, ["Trafo Kodu"])
    scada_col = find_col(raw.columns, ["SCADA-RTU Var mı?", "SCADA RTU Var mi", "SCADA"])
    osos_col = find_col(raw.columns, ["Trafo-OSOS Var mı?", "OSOS"])

    missing = []
    if montaj_col is None:
        missing.append("Montaj Yeri")
    if guc_col is None:
        missing.append("Gücü[kVA]")
    if tr_col is None:
        missing.append("Trafo Kodu")
    if scada_col is None:
        missing.append("SCADA-RTU Var mı?")

    if missing:
        raise ValueError(
            f"CBS dosyasında beklenen kolonlar yok: {missing}. Mevcut kolonlar: {list(raw.columns)}"
        )

    out = raw.copy()
    out["montaj_yeri"] = out[montaj_col].map(txt)
    out["montaj_key"] = out[montaj_col].apply(norm_key)
    out["tr_code"] = out[tr_col].apply(norm_tr)
    out["kurulu_guc_kva"] = pd.to_numeric(out[guc_col], errors="coerce")
    out["scada_rtu_var"] = out[scada_col].apply(yes)
    out["osos_var"] = out[osos_col].apply(yes) if osos_col else False

    asset_col = find_col(out.columns, ["AssetID", "Asset ID"])
    ilce_col = find_col(out.columns, ["İlçe", "Ilce"])
    mahalle_col = find_col(out.columns, ["Mahalle"])
    marka_col = find_col(out.columns, ["Marka"])
    tipi_col = find_col(out.columns, ["Tipi"])

    out["asset_id"] = out[asset_col].map(txt) if asset_col else ""
    out["ilce"] = out[ilce_col].map(txt) if ilce_col else ""
    out["mahalle"] = out[mahalle_col].map(txt) if mahalle_col else ""
    out["marka"] = out[marka_col].map(txt) if marka_col else ""
    out["tipi"] = out[tipi_col].map(txt) if tipi_col else ""

    lat_col = find_col(out.columns, ["lat", "latitude", "enlem", "y"])
    lon_col = find_col(out.columns, ["lon", "lng", "longitude", "boylam", "x"])

    out["lat"] = pd.to_numeric(out[lat_col], errors="coerce") if lat_col else np.nan
    out["lon"] = pd.to_numeric(out[lon_col], errors="coerce") if lon_col else np.nan

    out = out[out["montaj_key"].ne("") & out["tr_code"].ne("")]
    out = out.dropna(subset=["kurulu_guc_kva"])
    return out


def attach_cbs_capacity_to_scada(scada_clean: pd.DataFrame, cbs: pd.DataFrame) -> pd.DataFrame:
    cbs_scada = cbs[cbs["scada_rtu_var"]].copy()
    cbs_scada = cbs_scada.drop_duplicates(["montaj_key", "tr_code"], keep="first")

    keep_cols = [
        "montaj_key",
        "tr_code",
        "kurulu_guc_kva",
        "mahalle",
        "montaj_yeri",
        "asset_id",
    ]

    return scada_clean.merge(cbs_scada[keep_cols], on=["montaj_key", "tr_code"], how="left")


def filter_scada_data_errors(scada_with_cbs: pd.DataFrame):
    d = scada_with_cbs.copy()
    d["veri_hatasi_nedeni"] = ""

    d.loc[d["value"].isna(), "veri_hatasi_nedeni"] = "Boş demand"
    d.loc[d["value"] <= 0, "veri_hatasi_nedeni"] = "Sıfır veya negatif demand"
    d.loc[
        d["value"] > DATA_ERROR_ABS_MAX_KVA,
        "veri_hatasi_nedeni",
    ] = f"{DATA_ERROR_ABS_MAX_KVA} kVA üstü mantıksız demand"

    has_capacity = d["kurulu_guc_kva"].notna() & (d["kurulu_guc_kva"] > 0)
    too_high_vs_capacity = has_capacity & (
        d["value"] > d["kurulu_guc_kva"] * DATA_ERROR_CAP_MULTIPLIER
    )

    d.loc[
        too_high_vs_capacity,
        "veri_hatasi_nedeni",
    ] = f"Kurulu gücün {DATA_ERROR_CAP_MULTIPLIER} katından yüksek demand"

    errors = d[d["veri_hatasi_nedeni"].ne("")].copy()
    clean = d[d["veri_hatasi_nedeni"].eq("")].copy()

    return clean, errors


def aggregate_scada(clean_scada: pd.DataFrame) -> pd.DataFrame:
    if clean_scada.empty:
        return pd.DataFrame(
            columns=[
                "montaj_key",
                "tr_code",
                "timestamp",
                "demand_kva",
                "sample_count",
                "dm_label",
                "h_cell",
            ]
        )

    x = clean_scada.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], errors="coerce")
    x = x.dropna(subset=["timestamp"])
    x["period"] = x["timestamp"].dt.floor("h")

    hourly = (
        x.groupby(["montaj_key", "tr_code", "period"], as_index=False)
        .agg(
            demand_kva=("value", "max"),
            sample_count=("value", "size"),
            dm_label=("dm_label", "first"),
            h_cell=("h_cell", lambda s: ", ".join(sorted({txt(v) for v in s if txt(v)}))),
        )
        .rename(columns={"period": "timestamp"})
        .sort_values(["montaj_key", "tr_code", "timestamp"])
    )

    return hourly


def scada_metrics(hourly: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (montaj_key, tr_code), g in hourly.groupby(["montaj_key", "tr_code"]):
        y = g.sort_values("timestamp")
        s = y["demand_kva"].astype(float)
        max_idx = s.idxmax()

        rows.append(
            {
                "montaj_key": montaj_key,
                "tr_code": tr_code,
                "dm_label": y["dm_label"].iloc[-1] if "dm_label" in y else "",
                "h_cell": ", ".join(
                    sorted({txt(v) for v in y.get("h_cell", pd.Series(dtype=str)) if txt(v)})
                ),
                "veri_adedi": int(len(y)),
                "ilk_veri": y["timestamp"].min(),
                "son_veri": y["timestamp"].max(),
                "maksimum_demand_kva": float(s.max()),
                "maksimum_demand_zamani": y.loc[max_idx, "timestamp"],
                "ortalama_demand_kva": float(s.mean()),
                "p95_demand_kva": float(s.quantile(0.95)),
            }
        )

    return pd.DataFrame(rows)


def build_analysis(cbs: pd.DataFrame, hourly: pd.DataFrame, new_request_kva: float):
    cbs_scada = cbs[cbs["scada_rtu_var"]].copy()
    cbs_scada = cbs_scada.drop_duplicates(["montaj_key", "tr_code"], keep="first")

    metrics = scada_metrics(hourly)

    if metrics.empty:
        metrics = pd.DataFrame(columns=["montaj_key", "tr_code", "maksimum_demand_kva"])

    analysis = cbs_scada.merge(metrics, on=["montaj_key", "tr_code"], how="left")

    analysis["yuklenme_orani_pct"] = (
        analysis["maksimum_demand_kva"] / analysis["kurulu_guc_kva"] * 100
    )

    analysis["risk_seviyesi"] = analysis["yuklenme_orani_pct"].apply(risk_level)
    analysis["risk_sira"] = analysis["risk_seviyesi"].apply(risk_order_value)
    analysis["bos_kapasite_kva"] = analysis["kurulu_guc_kva"] - analysis["maksimum_demand_kva"]

    request = float(new_request_kva)

    analysis["yeni_talep_kva"] = request
    analysis["yeni_talep_sonrasi_demand_kva"] = analysis["maksimum_demand_kva"] + request
    analysis["yeni_talep_sonrasi_yuklenme_pct"] = (
        analysis["yeni_talep_sonrasi_demand_kva"] / analysis["kurulu_guc_kva"] * 100
    )
    analysis["yeni_talep_karari"] = analysis["yeni_talep_sonrasi_yuklenme_pct"].apply(
        connection_decision
    )

    if hourly.empty:
        scada_pairs = pd.DataFrame(columns=["montaj_key", "tr_code"])
    else:
        scada_pairs = hourly[["montaj_key", "tr_code"]].drop_duplicates()

    cbs_pairs = cbs_scada[["montaj_key", "tr_code"]].drop_duplicates()

    cbs_no_data = cbs_scada.merge(
        scada_pairs, on=["montaj_key", "tr_code"], how="left", indicator=True
    )
    cbs_no_data = cbs_no_data[cbs_no_data["_merge"].eq("left_only")].drop(columns=["_merge"])

    scada_no_cbs = scada_pairs.merge(
        cbs_pairs, on=["montaj_key", "tr_code"], how="left", indicator=True
    )
    scada_no_cbs = scada_no_cbs[scada_no_cbs["_merge"].eq("left_only")].drop(columns=["_merge"])

    if not scada_no_cbs.empty:
        scada_no_cbs = scada_no_cbs.merge(metrics, on=["montaj_key", "tr_code"], how="left")

    return analysis, cbs_no_data, scada_no_cbs


def recommend_transfer_candidates(analysis: pd.DataFrame, selected_key, transfer_kva: float) -> pd.DataFrame:
    selected = analysis[
        (analysis["montaj_key"].eq(selected_key[0]))
        & (analysis["tr_code"].eq(selected_key[1]))
    ]

    if selected.empty:
        return pd.DataFrame()

    row = selected.iloc[0]
    mahalle = txt(row["mahalle"])
    selected_load = row["yuklenme_orani_pct"]

    candidates = analysis.copy()

    candidates = candidates[
        candidates["mahalle"].astype(str).eq(mahalle)
        & ~(
            (candidates["montaj_key"].eq(row["montaj_key"]))
            & (candidates["tr_code"].eq(row["tr_code"]))
        )
        & candidates["maksimum_demand_kva"].notna()
        & candidates["yuklenme_orani_pct"].notna()
        & (candidates["yuklenme_orani_pct"] < selected_load)
        & (candidates["bos_kapasite_kva"] > 0)
    ].copy()

    transfer = float(transfer_kva)

    if transfer > 0:
        candidates = candidates[candidates["bos_kapasite_kva"] >= transfer]

    candidates["aktarim_sonrasi_demand_kva"] = candidates["maksimum_demand_kva"] + transfer
    candidates["aktarim_sonrasi_yuklenme_pct"] = (
        candidates["aktarim_sonrasi_demand_kva"] / candidates["kurulu_guc_kva"] * 100
    )
    candidates["aktarim_sonrasi_risk"] = candidates["aktarim_sonrasi_yuklenme_pct"].apply(
        risk_level
    )

    return candidates.sort_values(
        ["bos_kapasite_kva", "yuklenme_orani_pct"],
        ascending=[False, True],
    )


def hamule_recommendations(hourly: pd.DataFrame) -> pd.DataFrame:
    if hourly.empty:
        return pd.DataFrame()

    recs = []
    window = 2

    for (montaj_key, tr_code), g in hourly.groupby(["montaj_key", "tr_code"]):
        y = g.sort_values("timestamp").copy()
        s = y["demand_kva"].astype(float)

        roll_mean = s.rolling(window, min_periods=window).mean()
        roll_std = s.rolling(window, min_periods=window).std().fillna(0)

        y["score"] = roll_mean - 0.5 * roll_std

        top = y.dropna(subset=["score"]).sort_values("score", ascending=False).head(3).copy()

        if not top.empty:
            top["montaj_key"] = montaj_key
            top["tr_code"] = tr_code
            top["window_end"] = top["timestamp"]
            top["window_start"] = top["timestamp"] - pd.Timedelta(hours=window - 1)

            recs.append(
                top[
                    [
                        "montaj_key",
                        "tr_code",
                        "window_start",
                        "window_end",
                        "demand_kva",
                        "score",
                    ]
                ]
            )

    if recs:
        return pd.concat(recs, ignore_index=True)

    return pd.DataFrame()


def main():
    st.set_page_config(page_title="AYEDAŞ | Trafo Risk ve Yük Aktarım Paneli", layout="wide")

    st.title("⚡ CBS Entegre Mahalle Bazlı Trafo Risk Analizi ve Yük Aktarım Öneri Sistemi")
    st.caption(
        f"Sürüm: {APP_VERSION} | SCADA demand + CBS kurulu güç verisiyle risk analizi ve mahalle bazlı yük aktarım aday önerisi üretir."
    )

    scada_file = find_file(SCADA_FILE_CANDIDATES)
    cbs_file = find_file(CBS_FILE_CANDIDATES)

    with st.sidebar:
        st.header("Veri Kaynağı")
        st.write("SCADA:", scada_file.name if scada_file else "Bulunamadı")
        st.write("CBS:", cbs_file.name if cbs_file else "Bulunamadı")

        st.divider()
        st.header("Veri Temizleme")
        only_valid = st.toggle("Sadece Valid kalite", value=True)
        remove_zeros = st.toggle("0 demand değerlerini kaldır", value=True)
        st.caption(
            f"Veri hatası filtresi: {DATA_ERROR_ABS_MAX_KVA:,} kVA üstü veya kurulu gücün {DATA_ERROR_CAP_MULTIPLIER} katı üstü değerler elenir."
        )

        st.divider()
        st.header("Yeni Bağlantı")
        new_request_kva = st.number_input(
            "Yeni talep (kVA)",
            min_value=1.0,
            value=50.0,
            step=10.0,
        )

        st.divider()
        st.header("Risk Sınırları")
        st.caption("0-60%: Düşük Risk")
        st.caption("60-80%: Orta Risk")
        st.caption("80-100%: Yüksek Risk")
        st.caption("100% üzeri: Kritik Risk")

    if not scada_file:
        st.error("SCADA Excel dosyası bulunamadı. Dosya adını repo kökünde 'Sancaktepe Trafo demand 2025.xlsx' yap.")
        st.stop()

    if not cbs_file:
        st.error("CBS Excel dosyası bulunamadı. Dosya adını repo kökünde 'Trafo Sorgu Sonuçları.xlsx' yap.")
        st.stop()

    try:
        scada_raw = load_scada_excel(str(scada_file))
        cbs_raw = load_cbs_excel(str(cbs_file))

        scada_prepared = prepare_scada(
            scada_raw,
            only_valid=only_valid,
            remove_zeros=remove_zeros,
        )

        scada_with_cbs = attach_cbs_capacity_to_scada(scada_prepared, cbs_raw)
        scada_clean, scada_errors = filter_scada_data_errors(scada_with_cbs)
        hourly = aggregate_scada(scada_clean)

        analysis, cbs_no_data, scada_no_cbs = build_analysis(
            cbs_raw,
            hourly,
            new_request_kva,
        )

    except Exception as e:
        st.error(f"Dosyalar okunurken hata oluştu: {type(e).__name__}: {e}")
        with st.expander("Teknik hata detayı"):
            st.exception(e)
        st.stop()

    matched = analysis[analysis["veri_adedi"].notna()].copy()
    critical = matched[matched["risk_seviyesi"].eq("Kritik Risk")].copy()
    high_or_critical = matched[
        matched["risk_seviyesi"].isin(["Yüksek Risk", "Kritik Risk"])
    ].copy()
    investment_needed = analysis[analysis["yeni_talep_karari"].eq("Yatırım Gerekli")].copy()

    k1, k2, k3, k4, k5, k6 = st.columns(6)

    k1.metric("CBS SCADA-RTU = Evet", f"{len(analysis):,}")
    k2.metric("Eşleşen Trafo", f"{len(matched):,}")
    k3.metric("Yüksek + Kritik", f"{len(high_or_critical):,}")
    k4.metric("Kritik Risk", f"{len(critical):,}")
    k5.metric("Filtrelenen Veri Hatası", f"{len(scada_errors):,}")
    k6.metric("Yeni Talepte Yatırım", f"{len(investment_needed):,}")

    tab_risk, tab_map, tab_transfer, tab_connection, tab_detail, tab_quality = st.tabs(
        [
            "📊 Riskli Trafolar",
            "🗺️ CBS Harita",
            "🔁 Yük Aktarım Önerisi",
            "🧮 Yeni Bağlantı",
            "🔍 Trafo Detay",
            "🧹 Veri Kalitesi",
        ]
    )

    main_cols = [
        "montaj_yeri",
        "tr_code",
        "mahalle",
        "kurulu_guc_kva",
        "maksimum_demand_kva",
        "maksimum_demand_zamani",
        "yuklenme_orani_pct",
        "bos_kapasite_kva",
        "risk_seviyesi",
    ]

    main_rename = {
        "montaj_yeri": "Montaj Yeri",
        "tr_code": "Trafo Kodu",
        "mahalle": "Mahalle",
        "kurulu_guc_kva": "Trafo Gücü (kVA)",
        "maksimum_demand_kva": "Maksimum Demand (kVA)",
        "maksimum_demand_zamani": "Maksimum Demand Zamanı",
        "yuklenme_orani_pct": "Yüklenme Oranı (%)",
        "bos_kapasite_kva": "Boş Kapasite (kVA)",
        "risk_seviyesi": "Risk Seviyesi",
    }

    with tab_risk:
        st.subheader("Riskli trafolar listesi")
        st.info(
            "Yüklenme Oranı = Maksimum Demand / Trafo Gücü × 100. Veri hatası olan SCADA ölçümleri bu hesaba girmeden elenir."
        )

        f1, f2, f3 = st.columns([1, 1, 2])

        with f1:
            risk_filter = st.multiselect(
                "Risk seviyesi",
                [
                    "Kritik Risk",
                    "Yüksek Risk",
                    "Orta Risk",
                    "Düşük Risk",
                    "SCADA Verisi Yok",
                ],
                default=[
                    "Kritik Risk",
                    "Yüksek Risk",
                    "Orta Risk",
                    "Düşük Risk",
                    "SCADA Verisi Yok",
                ],
            )

        with f2:
            mahalleler = sorted(
                [m for m in analysis["mahalle"].dropna().astype(str).unique() if m]
            )
            mahalle_filter = st.multiselect("Mahalle", mahalleler)

        with f3:
            search = st.text_input("Montaj yeri / AssetID / Trafo kodu ara")

        view = analysis.copy()

        if risk_filter:
            view = view[view["risk_seviyesi"].isin(risk_filter)]

        if mahalle_filter:
            view = view[view["mahalle"].isin(mahalle_filter)]

        if txt(search):
            s = txt(search).upper()
            view = view[
                view["montaj_yeri"].astype(str).str.upper().str.contains(s, na=False)
                | view["asset_id"].astype(str).str.upper().str.contains(s, na=False)
                | view["tr_code"].astype(str).str.upper().str.contains(s, na=False)
            ]

        view = view.sort_values(
            ["risk_sira", "yuklenme_orani_pct"],
            ascending=[False, False],
            na_position="last",
        )

        st.dataframe(
            view[main_cols].rename(columns=main_rename),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Trafo Gücü (kVA)": st.column_config.NumberColumn(format="%.0f"),
                "Maksimum Demand (kVA)": st.column_config.NumberColumn(format="%.2f"),
                "Yüklenme Oranı (%)": st.column_config.ProgressColumn(
                    min_value=0,
                    max_value=150,
                    format="%.1f%%",
                ),
                "Boş Kapasite (kVA)": st.column_config.NumberColumn(format="%.2f"),
            },
        )

        csv = view[main_cols].rename(columns=main_rename).to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Risk analizini CSV indir",
            csv,
            "trafo_risk_analizi.csv",
            "text/csv",
        )

        chart = view.dropna(subset=["yuklenme_orani_pct"]).head(20).copy()

        if not chart.empty:
            chart["Etiket"] = (
                chart["montaj_yeri"].astype(str) + " / " + chart["tr_code"].astype(str)
            )

            fig = px.bar(
                chart.sort_values("yuklenme_orani_pct"),
                x="yuklenme_orani_pct",
                y="Etiket",
                orientation="h",
                color="risk_seviyesi",
                title="En yüksek yüklenme oranına sahip ilk 20 trafo",
                labels={
                    "yuklenme_orani_pct": "Yüklenme Oranı (%)",
                    "risk_seviyesi": "Risk",
                },
            )

            fig.add_vline(x=60, line_dash="dash", annotation_text="%60")
            fig.add_vline(x=80, line_dash="dash", annotation_text="%80")
            fig.add_vline(x=100, line_dash="dash", annotation_text="%100")

            st.plotly_chart(fig, use_container_width=True)

    with tab_map:
        st.subheader("CBS harita gösterimi")

        map_data = analysis.dropna(subset=["lat", "lon", "yuklenme_orani_pct"]).copy()

        if map_data.empty:
            st.warning(
                "Bu CBS dosyasında koordinat kolonu bulunmadığı için harita çizilemiyor. Enlem/Boylam veya X/Y kolonları eklenirse bu modül otomatik çalışır."
            )
        else:
            fig = px.scatter_mapbox(
                map_data,
                lat="lat",
                lon="lon",
                color="risk_seviyesi",
                size="yuklenme_orani_pct",
                hover_name="montaj_yeri",
                hover_data={
                    "tr_code": True,
                    "mahalle": True,
                    "kurulu_guc_kva": True,
                    "maksimum_demand_kva": True,
                    "yuklenme_orani_pct": ":.1f",
                    "lat": False,
                    "lon": False,
                },
                zoom=11,
                height=650,
            )

            fig.update_layout(
                mapbox_style="open-street-map",
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
            )

            st.plotly_chart(fig, use_container_width=True)

    with tab_transfer:
        st.subheader("Mahalle bazlı yük aktarım öneri modülü")
        st.info(
            "Sistem doğrudan yük aktarımı kararı vermez. Aynı mahallede daha düşük yüklenmiş ve boş kapasitesi olan trafoları aday olarak sıralar."
        )

        selectable = matched[
            matched["risk_seviyesi"].isin(["Kritik Risk", "Yüksek Risk"])
        ].copy()

        if selectable.empty:
            st.success("Yüksek veya kritik riskli trafo bulunamadı.")
        else:
            selectable = selectable.sort_values("yuklenme_orani_pct", ascending=False)

            selectable["secim"] = (
                selectable["montaj_yeri"].astype(str)
                + " / "
                + selectable["tr_code"].astype(str)
                + " | "
                + selectable["mahalle"].astype(str)
                + " | %"
                + selectable["yuklenme_orani_pct"].round(1).astype(str)
            )

            selected_label = st.selectbox("Riskli trafo seç", selectable["secim"].tolist())
            selected_row = selectable[selectable["secim"].eq(selected_label)].iloc[0]

            overload_kva = max(
                0.0,
                float(selected_row["maksimum_demand_kva"] - selected_row["kurulu_guc_kva"]),
            )

            default_transfer = round(overload_kva, 2) if overload_kva > 0 else 50.0

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Seçili Trafo",
                f"{selected_row['montaj_yeri']} / {selected_row['tr_code']}",
            )
            c2.metric("Mahalle", selected_row["mahalle"])
            c3.metric("Yüklenme", fmt(selected_row["yuklenme_orani_pct"], 1, "%"))
            c4.metric("100% Üstü Fazla Yük", fmt(overload_kva, 2, " kVA"))

            transfer_kva = st.number_input(
                "Aktarılması düşünülen yük (kVA)",
                min_value=0.0,
                value=float(default_transfer),
                step=10.0,
                help="Kritik trafodan başka bir trafoya aktarılması düşünülen yaklaşık yük. Aday trafonun boş kapasitesi bu değerden büyük olmalı.",
            )

            candidates = recommend_transfer_candidates(
                analysis,
                selected_key=(selected_row["montaj_key"], selected_row["tr_code"]),
                transfer_kva=transfer_kva,
            )

            if candidates.empty:
                st.warning("Aynı mahallede bu aktarım miktarı için uygun aday trafo bulunamadı.")
            else:
                candidate_cols = [
                    "montaj_yeri",
                    "tr_code",
                    "mahalle",
                    "kurulu_guc_kva",
                    "maksimum_demand_kva",
                    "yuklenme_orani_pct",
                    "bos_kapasite_kva",
                    "aktarim_sonrasi_demand_kva",
                    "aktarim_sonrasi_yuklenme_pct",
                    "aktarim_sonrasi_risk",
                ]

                candidate_rename = {
                    "montaj_yeri": "Aday Montaj Yeri",
                    "tr_code": "Aday Trafo",
                    "mahalle": "Mahalle",
                    "kurulu_guc_kva": "Trafo Gücü (kVA)",
                    "maksimum_demand_kva": "Mevcut Maks. Demand (kVA)",
                    "yuklenme_orani_pct": "Mevcut Yüklenme (%)",
                    "bos_kapasite_kva": "Boş Kapasite (kVA)",
                    "aktarim_sonrasi_demand_kva": "Aktarım Sonrası Demand (kVA)",
                    "aktarim_sonrasi_yuklenme_pct": "Aktarım Sonrası Yüklenme (%)",
                    "aktarim_sonrasi_risk": "Aktarım Sonrası Risk",
                }

                st.dataframe(
                    candidates[candidate_cols].rename(columns=candidate_rename),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Mevcut Yüklenme (%)": st.column_config.ProgressColumn(
                            min_value=0,
                            max_value=150,
                            format="%.1f%%",
                        ),
                        "Aktarım Sonrası Yüklenme (%)": st.column_config.ProgressColumn(
                            min_value=0,
                            max_value=150,
                            format="%.1f%%",
                        ),
                    },
                )

                st.warning(
                    "Öneriler karar destek amaçlıdır. AG şebeke bağlantıları, kablo güzergahları, faz dengesi, gerilim düşümü ve saha uygunluğu kontrol edilmeden gerçek yük aktarımı yapılmamalıdır."
                )

    with tab_connection:
        st.subheader("Yeni bağlantı talebi simülasyonu")
        st.write(f"Yeni talep doğrudan kVA girilir. Seçilen talep: **{new_request_kva:.2f} kVA**")

        sim_cols = [
            "montaj_yeri",
            "tr_code",
            "mahalle",
            "kurulu_guc_kva",
            "maksimum_demand_kva",
            "yuklenme_orani_pct",
            "yeni_talep_kva",
            "yeni_talep_sonrasi_demand_kva",
            "yeni_talep_sonrasi_yuklenme_pct",
            "yeni_talep_karari",
        ]

        sim_rename = {
            "montaj_yeri": "Montaj Yeri",
            "tr_code": "Trafo Kodu",
            "mahalle": "Mahalle",
            "kurulu_guc_kva": "Trafo Gücü (kVA)",
            "maksimum_demand_kva": "Maksimum Demand (kVA)",
            "yuklenme_orani_pct": "Mevcut Yüklenme (%)",
            "yeni_talep_kva": "Yeni Talep (kVA)",
            "yeni_talep_sonrasi_demand_kva": "Yeni Talep Sonrası Demand (kVA)",
            "yeni_talep_sonrasi_yuklenme_pct": "Yeni Talep Sonrası Yüklenme (%)",
            "yeni_talep_karari": "Karar",
        }

        sim = analysis.sort_values(
            "yeni_talep_sonrasi_yuklenme_pct",
            ascending=False,
            na_position="last",
        )

        st.dataframe(
            sim[sim_cols].rename(columns=sim_rename),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Mevcut Yüklenme (%)": st.column_config.ProgressColumn(
                    min_value=0,
                    max_value=150,
                    format="%.1f%%",
                ),
                "Yeni Talep Sonrası Yüklenme (%)": st.column_config.ProgressColumn(
                    min_value=0,
                    max_value=150,
                    format="%.1f%%",
                ),
            },
        )

    with tab_detail:
        st.subheader("Trafo detay")

        if matched.empty:
            st.warning("Eşleşen trafo yok. Veri Kalitesi sekmesini kontrol et.")
        else:
            options = matched.sort_values(["montaj_yeri", "tr_code"]).copy()
            options["secim"] = (
                options["montaj_yeri"].astype(str) + " / " + options["tr_code"].astype(str)
            )

            selected = st.selectbox("Trafo seç", options["secim"].tolist())
            row = options[options["secim"].eq(selected)].iloc[0]

            ts = hourly[
                hourly["montaj_key"].eq(row["montaj_key"])
                & hourly["tr_code"].eq(row["tr_code"])
            ].sort_values("timestamp")

            d1, d2, d3, d4, d5 = st.columns(5)

            d1.metric("Trafo Gücü", fmt(row["kurulu_guc_kva"], 0, " kVA"))
            d2.metric("Maks Demand", fmt(row["maksimum_demand_kva"], 2, " kVA"))
            d3.metric("Yüklenme", fmt(row["yuklenme_orani_pct"], 1, "%"))
            d4.metric("Boş Kapasite", fmt(row["bos_kapasite_kva"], 2, " kVA"))
            d5.metric("Risk", row["risk_seviyesi"])

            st.write(
                f"**Mahalle:** {row['mahalle']} | "
                f"**AssetID:** {row['asset_id']} | "
                f"**SCADA:** {row.get('dm_label', '')} / {row.get('h_cell', '')}"
            )

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=ts["timestamp"],
                    y=ts["demand_kva"],
                    mode="lines+markers",
                    name="Demand",
                )
            )

            fig.add_hline(
                y=row["kurulu_guc_kva"] * 0.60,
                line_dash="dash",
                annotation_text="%60",
            )
            fig.add_hline(
                y=row["kurulu_guc_kva"] * 0.80,
                line_dash="dash",
                annotation_text="%80",
            )
            fig.add_hline(
                y=row["kurulu_guc_kva"],
                line_dash="dash",
                annotation_text="%100",
            )

            fig.update_layout(
                title=f"{row['montaj_yeri']} / {row['tr_code']} Demand Profili",
                xaxis_title="Zaman",
                yaxis_title="kVA",
            )

            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(ts, use_container_width=True, hide_index=True)

    with tab_quality:
        st.subheader("Veri kalitesi ve eşleşme kontrolü")

        q1, q2, q3, q4, q5 = st.columns(5)

        q1.metric("SCADA Ham Kayıt", f"{len(scada_raw):,}")
        q2.metric("SCADA Hazırlanmış", f"{len(scada_prepared):,}")
        q3.metric("Filtrelenen Veri Hatası", f"{len(scada_errors):,}")
        q4.metric("CBS Toplam Trafo", f"{len(cbs_raw):,}")
        q5.metric("CBS SCADA-RTU = Evet", f"{int(cbs_raw['scada_rtu_var'].sum()):,}")

        st.markdown("#### Filtrelenen SCADA veri hataları")

        if scada_errors.empty:
            st.success("Veri hatası olarak elenen kayıt yok.")
        else:
            error_cols = [
                "timestamp",
                "point_name",
                "value",
                "kurulu_guc_kva",
                "montaj_yeri",
                "tr_code",
                "veri_hatasi_nedeni",
            ]

            error_rename = {
                "timestamp": "Zaman",
                "point_name": "SCADA Point Name",
                "value": "Demand Değeri",
                "kurulu_guc_kva": "Trafo Gücü (kVA)",
                "montaj_yeri": "Montaj Yeri",
                "tr_code": "Trafo Kodu",
                "veri_hatasi_nedeni": "Elenme Nedeni",
            }

            st.dataframe(
                scada_errors[error_cols].rename(columns=error_rename).head(500),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("#### CBS'de SCADA-RTU = Evet ama SCADA demand eşleşmeyenler")

        if cbs_no_data.empty:
            st.success("Kayıt yok.")
        else:
            cols = [
                "montaj_yeri",
                "tr_code",
                "asset_id",
                "mahalle",
                "kurulu_guc_kva",
            ]

            st.dataframe(
                cbs_no_data[cols].rename(columns=main_rename),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("#### SCADA demand var ama CBS SCADA-RTU = Evet listesinde karşılığı olmayanlar")

        if scada_no_cbs.empty:
            st.success("Kayıt yok.")
        else:
            st.dataframe(scada_no_cbs, use_container_width=True, hide_index=True)

        with st.expander("Eşleşme ve hesaplama mantığı"):
            st.write(
                "SCADA `T-4092` → CBS `T4092`; "
                "SCADA `RP_4004 DM` veya `4004 DM` → CBS `B4004`; "
                "SCADA `Enan1/Enan2` → CBS `TR1/TR2`. "
                "H01/H03 gibi alanlar trafo kodu değil, hücre/nokta bilgisidir. "
                "Risk hesabı maksimum demand ile yapılır. Veri hatası filtresinden geçen maksimum demand, trafo gücüne bölünerek yüklenme oranı hesaplanır."
            )


if __name__ == "__main__":
    main()
