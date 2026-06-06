import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_VERSION = "v4-float-fix-sade"

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

WARN_PCT = 80
CRITICAL_PCT = 95
PLANNING_QUANTILE = 0.95


def txt(x) -> str:
    """
    Her değeri güvenli metne çevirir.
    Float/NaN hücrelerde 'in' araması yapılmasını engeller.
    """
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
    tr_map = str.maketrans("çğıöşüİ", "cgiosui")
    s = s.translate(tr_map)
    return re.sub(r"[^a-z0-9]+", "", s)


def norm_key(x) -> str:
    return re.sub(r"[^A-Z0-9]", "", txt(x).upper())


def norm_tr(x) -> str:
    s = txt(x).upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
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
        a = norm_col(alt)
        if a in normalized:
            return normalized[a]

    for c in columns:
        c_norm = norm_col(c)
        for alt in alternatives:
            a = norm_col(alt)
            if a and a in c_norm:
                return c

    return None


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
            f"SCADA dosyasında beklenen kolonlar yok: {missing}. "
            f"Mevcut kolonlar: {list(df.columns)}"
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
    """
    SCADA örnekleri:
    /Net-E/SANCAKTEPE OM DTM/T-4092/0.4kV/Enan2 H01/S
    /Net-E/SANCAKTEPE OM/RP_4004 DM/0.4kV/Enan1 H01/S
    /Net-E/SANCAKTEPE OM/4100 DM/0.4kV/Enan1 H19/S

    Eşleşme:
    T-4092     -> T4092
    RP_4004 DM -> B4004
    4100 DM    -> B4100
    Enan1      -> TR1
    Enan2      -> TR2
    """
    p = txt(point_name)
    parts = p.split("/")

    station_part = ""
    for i, part in enumerate(parts):
        if "0.4" in txt(part).lower() and i > 0:
            station_part = txt(parts[i - 1])
            break

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


def aggregate_scada(d: pd.DataFrame) -> pd.DataFrame:
    if d.empty:
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

    x = d.copy()
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


@st.cache_data(show_spinner=False)
def load_cbs_excel(path: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=0)
    raw.columns = [txt(c) for c in raw.columns]

    montaj_col = find_col(raw.columns, ["Montaj Yeri"])
    guc_col = find_col(raw.columns, ["Gücü[kVA]", "Gucu[kVA]", "Gücü", "Gucu"])
    tr_col = find_col(raw.columns, ["Trafo Kodu"])
    scada_col = find_col(raw.columns, ["SCADA-RTU Var mı?", "SCADA RTU Var mi", "SCADA"])

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
            f"CBS dosyasında beklenen kolonlar yok: {missing}. "
            f"Mevcut kolonlar: {list(raw.columns)}"
        )

    out = raw.copy()
    out["montaj_yeri"] = out[montaj_col].map(txt)
    out["montaj_key"] = out[montaj_col].apply(norm_key)
    out["tr_code"] = out[tr_col].apply(norm_tr)
    out["kurulu_guc_kva"] = pd.to_numeric(out[guc_col], errors="coerce")
    out["scada_rtu_var"] = out[scada_col].apply(yes)

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

    out = out[out["montaj_key"].ne("") & out["tr_code"].ne("")]
    out = out.dropna(subset=["kurulu_guc_kva"])

    return out


def scada_metrics(hourly: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (montaj_key, tr_code), g in hourly.groupby(["montaj_key", "tr_code"]):
        y = g.sort_values("timestamp")
        s = y["demand_kva"].astype(float)

        rows.append(
            {
                "montaj_key": montaj_key,
                "tr_code": tr_code,
                "dm_label": y["dm_label"].iloc[-1] if "dm_label" in y else "",
                "h_cell": ", ".join(sorted({txt(v) for v in y.get("h_cell", pd.Series(dtype=str)) if txt(v)})),
                "veri_adedi": int(len(y)),
                "ilk_veri": y["timestamp"].min(),
                "son_veri": y["timestamp"].max(),
                "son_demand_kva": float(s.iloc[-1]),
                "maks_demand_kva": float(s.max()),
                "ortalama_demand_kva": float(s.mean()),
                "p95_demand_kva": float(s.quantile(PLANNING_QUANTILE)),
            }
        )

    return pd.DataFrame(rows)


def risk_status(load_pct):
    if pd.isna(load_pct):
        return "SCADA Verisi Yok"
    if load_pct >= 100:
        return "Aşırı Yüklü"
    if load_pct >= CRITICAL_PCT:
        return "Kritik"
    if load_pct >= WARN_PCT:
        return "Sınırda"
    return "Normal"


def connection_status(projected_pct):
    if pd.isna(projected_pct):
        return "SCADA Verisi Yok"
    if projected_pct >= 100:
        return "Yatırım Gerekli"
    if projected_pct >= CRITICAL_PCT:
        return "Kritik İnceleme"
    if projected_pct >= WARN_PCT:
        return "Şartlı Uygun / İzlenmeli"
    return "Uygun"


def build_analysis(cbs: pd.DataFrame, hourly: pd.DataFrame, demand_kw: float, pf: float):
    cbs_scada = cbs[cbs["scada_rtu_var"]].copy()
    cbs_scada = cbs_scada.drop_duplicates(subset=["montaj_key", "tr_code"], keep="first")

    m = scada_metrics(hourly)
    if m.empty:
        m = pd.DataFrame(columns=["montaj_key", "tr_code", "p95_demand_kva"])

    analysis = cbs_scada.merge(m, on=["montaj_key", "tr_code"], how="left")

    analysis["yuklenme_pct"] = analysis["p95_demand_kva"] / analysis["kurulu_guc_kva"] * 100
    analysis["risk_durumu"] = analysis["yuklenme_pct"].apply(risk_status)

    analysis["bos_kapasite_80_kva"] = analysis["kurulu_guc_kva"] * WARN_PCT / 100 - analysis["p95_demand_kva"]
    analysis["bos_kapasite_100_kva"] = analysis["kurulu_guc_kva"] - analysis["p95_demand_kva"]

    request_kva = float(demand_kw) / max(float(pf), 0.01)
    analysis["yeni_talep_kw"] = float(demand_kw)
    analysis["yeni_talep_kva"] = request_kva
    analysis["projeksiyon_demand_kva"] = analysis["p95_demand_kva"] + request_kva
    analysis["projeksiyon_yuklenme_pct"] = analysis["projeksiyon_demand_kva"] / analysis["kurulu_guc_kva"] * 100
    analysis["baglanti_karari"] = analysis["projeksiyon_yuklenme_pct"].apply(connection_status)

    if hourly.empty:
        scada_pairs = pd.DataFrame(columns=["montaj_key", "tr_code"])
    else:
        scada_pairs = hourly[["montaj_key", "tr_code"]].drop_duplicates()

    cbs_pairs = cbs_scada[["montaj_key", "tr_code"]].drop_duplicates()

    cbs_no_data = cbs_scada.merge(scada_pairs, on=["montaj_key", "tr_code"], how="left", indicator=True)
    cbs_no_data = cbs_no_data[cbs_no_data["_merge"].eq("left_only")].drop(columns=["_merge"])

    scada_no_cbs = scada_pairs.merge(cbs_pairs, on=["montaj_key", "tr_code"], how="left", indicator=True)
    scada_no_cbs = scada_no_cbs[scada_no_cbs["_merge"].eq("left_only")].drop(columns=["_merge"])

    if not scada_no_cbs.empty:
        scada_no_cbs = scada_no_cbs.merge(m, on=["montaj_key", "tr_code"], how="left")

    return analysis, cbs_no_data, scada_no_cbs


def fmt(x, digits=1, suffix=""):
    if pd.isna(x):
        return "-"
    return f"{float(x):,.{digits}f}{suffix}"


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

        score = roll_mean - 0.5 * roll_std
        y["score"] = score

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

    return pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()


def main():
    st.set_page_config(page_title="AYEDAŞ | Trafo Kapasite Paneli", layout="wide")

    st.title("⚡ AYEDAŞ Trafo Kapasite & Demand Karar Destek Paneli")
    st.caption(
        f"Sürüm: {APP_VERSION} | "
        "CBS'de SCADA-RTU = Evet olan trafolar, SCADA demand verisiyle karşılaştırılır."
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

        st.divider()
        st.header("Yeni Bağlantı")
        demand_kw = st.number_input("Yeni talep (kW)", min_value=1.0, value=50.0, step=10.0)
        pf = st.number_input("Güç faktörü", min_value=0.50, max_value=1.00, value=0.90, step=0.01)

        st.divider()
        st.header("Hesaplama Notları")
        st.caption("Ana demand metriği: P95 demand")
        st.caption("Sınırda: %80 ve üzeri")
        st.caption("Kritik: %95 ve üzeri")
        st.caption("Aşırı yüklü: %100 ve üzeri")

    if not scada_file:
        st.error("SCADA Excel dosyası bulunamadı. Dosya adını repo kökünde 'Sancaktepe Trafo demand 2025.xlsx' yap.")
        st.stop()

    if not cbs_file:
        st.error("CBS Excel dosyası bulunamadı. Dosya adını repo kökünde 'Trafo Sorgu Sonuçları.xlsx' yap.")
        st.stop()

    try:
        scada_raw = load_scada_excel(str(scada_file))
        cbs_raw = load_cbs_excel(str(cbs_file))

        scada_clean = prepare_scada(scada_raw, only_valid=only_valid, remove_zeros=remove_zeros)
        hourly = aggregate_scada(scada_clean)

        analysis, cbs_no_data, scada_no_cbs = build_analysis(cbs_raw, hourly, demand_kw, pf)

    except Exception as e:
        st.error(f"Dosyalar okunurken hata oluştu: {type(e).__name__}: {e}")
        with st.expander("Teknik hata detayı"):
            st.exception(e)
        st.stop()

    matched = analysis[analysis["veri_adedi"].notna()].copy()
    risky = matched[matched["risk_durumu"].isin(["Sınırda", "Kritik", "Aşırı Yüklü"])]
    invest = analysis[analysis["baglanti_karari"].eq("Yatırım Gerekli")]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("CBS SCADA-RTU = Evet", f"{len(analysis):,}")
    k2.metric("SCADA temiz kayıt", f"{len(scada_clean):,}")
    k3.metric("Eşleşen trafo", f"{len(matched):,}")
    k4.metric("Riskli trafo", f"{len(risky):,}")
    k5.metric("Yeni talepte yatırım", f"{len(invest):,}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Kapasite Analizi",
            "🔍 Trafo Detay",
            "🧮 Yeni Bağlantı Simülasyonu",
            "🎯 Hamule Ölçüm Önerisi",
            "🧹 Veri Kalitesi",
        ]
    )

    base_cols = [
        "montaj_yeri",
        "tr_code",
        "asset_id",
        "ilce",
        "mahalle",
        "kurulu_guc_kva",
        "p95_demand_kva",
        "yuklenme_pct",
        "bos_kapasite_80_kva",
        "bos_kapasite_100_kva",
        "risk_durumu",
        "veri_adedi",
        "son_veri",
    ]

    with tab1:
        st.subheader("Kapasite analizi")
        st.info(
            "Bu tablo sadece CBS'de SCADA-RTU = Evet olan trafoları gösterir. "
            "Karar metriği P95 demand / kurulu güç oranıdır."
        )

        c1, c2, c3 = st.columns([1, 1, 2])

        with c1:
            risk_filter = st.multiselect(
                "Risk durumu",
                ["Normal", "Sınırda", "Kritik", "Aşırı Yüklü", "SCADA Verisi Yok"],
                default=["Normal", "Sınırda", "Kritik", "Aşırı Yüklü", "SCADA Verisi Yok"],
            )

        with c2:
            mahalleler = sorted([m for m in analysis["mahalle"].dropna().astype(str).unique() if m])
            mahalle_filter = st.multiselect("Mahalle", mahalleler)

        with c3:
            search = st.text_input("Montaj yeri / AssetID / Trafo kodu ara")

        view = analysis.copy()

        if risk_filter:
            view = view[view["risk_durumu"].isin(risk_filter)]

        if mahalle_filter:
            view = view[view["mahalle"].isin(mahalle_filter)]

        if txt(search):
            s = txt(search).upper()
            view = view[
                view["montaj_yeri"].astype(str).str.upper().str.contains(s, na=False)
                | view["asset_id"].astype(str).str.upper().str.contains(s, na=False)
                | view["tr_code"].astype(str).str.upper().str.contains(s, na=False)
            ]

        view = view.sort_values("yuklenme_pct", ascending=False, na_position="last")

        st.dataframe(
            view[base_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "montaj_yeri": "Montaj Yeri",
                "tr_code": "Trafo Kodu",
                "kurulu_guc_kva": st.column_config.NumberColumn("Kurulu Güç (kVA)", format="%.0f"),
                "p95_demand_kva": st.column_config.NumberColumn("P95 Demand (kVA)", format="%.2f"),
                "yuklenme_pct": st.column_config.ProgressColumn(
                    "Yüklenme %",
                    min_value=0,
                    max_value=120,
                    format="%.1f%%",
                ),
                "bos_kapasite_80_kva": st.column_config.NumberColumn("80% Eşiğe Boşluk", format="%.2f"),
                "bos_kapasite_100_kva": st.column_config.NumberColumn("100% Eşiğe Boşluk", format="%.2f"),
            },
        )

        csv = view[base_cols].to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Kapasite analizini CSV indir", csv, "kapasite_analizi.csv", "text/csv")

        chart = view.dropna(subset=["yuklenme_pct"]).head(20).copy()

        if not chart.empty:
            chart["etiket"] = chart["montaj_yeri"].astype(str) + " / " + chart["tr_code"].astype(str)

            fig = px.bar(
                chart.sort_values("yuklenme_pct"),
                x="yuklenme_pct",
                y="etiket",
                orientation="h",
                title="En yüksek yüklenme oranına sahip ilk 20 trafo",
            )

            fig.add_vline(x=WARN_PCT, line_dash="dash", annotation_text="%80")
            fig.add_vline(x=CRITICAL_PCT, line_dash="dash", annotation_text="%95")
            fig.update_layout(xaxis_title="Yüklenme (%)", yaxis_title="Trafo")

            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Trafo detay")

        if matched.empty:
            st.warning("Eşleşen trafo yok. Veri Kalitesi sekmesini kontrol et.")
        else:
            options = matched.sort_values(["montaj_yeri", "tr_code"]).copy()
            options["secim"] = options["montaj_yeri"].astype(str) + " / " + options["tr_code"].astype(str)

            selected = st.selectbox("Trafo seç", options["secim"].tolist())
            row = options[options["secim"].eq(selected)].iloc[0]

            ts = hourly[
                (hourly["montaj_key"].eq(row["montaj_key"]))
                & (hourly["tr_code"].eq(row["tr_code"]))
            ].sort_values("timestamp")

            a, b, c, d, e = st.columns(5)

            a.metric("Kurulu güç", fmt(row["kurulu_guc_kva"], 0, " kVA"))
            b.metric("P95 demand", fmt(row["p95_demand_kva"], 2, " kVA"))
            c.metric("Maks demand", fmt(row["maks_demand_kva"], 2, " kVA"))
            d.metric("Yüklenme", fmt(row["yuklenme_pct"], 1, "%"))
            e.metric("Risk", row["risk_durumu"])

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
                y=row["kurulu_guc_kva"] * WARN_PCT / 100,
                line_dash="dash",
                annotation_text="%80",
            )
            fig.add_hline(
                y=row["kurulu_guc_kva"] * CRITICAL_PCT / 100,
                line_dash="dash",
                annotation_text="%95",
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

    with tab3:
        st.subheader("Yeni bağlantı simülasyonu")

        st.write(
            f"Yeni talep: **{demand_kw:.0f} kW** | "
            f"Güç faktörü: **{pf:.2f}** | "
            f"kVA karşılığı: **{demand_kw / pf:.2f} kVA**"
        )

        sim_cols = [
            "montaj_yeri",
            "tr_code",
            "mahalle",
            "kurulu_guc_kva",
            "p95_demand_kva",
            "yuklenme_pct",
            "yeni_talep_kva",
            "projeksiyon_demand_kva",
            "projeksiyon_yuklenme_pct",
            "baglanti_karari",
        ]

        sim = analysis.sort_values("projeksiyon_yuklenme_pct", ascending=False, na_position="last")

        st.dataframe(
            sim[sim_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "yuklenme_pct": st.column_config.ProgressColumn(
                    "Mevcut %",
                    min_value=0,
                    max_value=120,
                    format="%.1f%%",
                ),
                "projeksiyon_yuklenme_pct": st.column_config.ProgressColumn(
                    "Yeni Talep Sonrası %",
                    min_value=0,
                    max_value=120,
                    format="%.1f%%",
                ),
            },
        )

        csv = sim[sim_cols].to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Simülasyonu CSV indir", csv, "baglanti_simulasyonu.csv", "text/csv")

    with tab4:
        st.subheader("Hamule ölçümü için önerilen zaman pencereleri")

        recs = hamule_recommendations(hourly)

        if recs.empty:
            st.warning("Öneri üretilemedi.")
        else:
            recs = recs.merge(
                cbs_raw[
                    [
                        "montaj_key",
                        "tr_code",
                        "montaj_yeri",
                        "mahalle",
                        "kurulu_guc_kva",
                    ]
                ].drop_duplicates(["montaj_key", "tr_code"]),
                on=["montaj_key", "tr_code"],
                how="left",
            )

            cols = [
                "montaj_yeri",
                "tr_code",
                "mahalle",
                "window_start",
                "window_end",
                "demand_kva",
                "score",
            ]

            st.dataframe(
                recs[cols].sort_values("score", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

    with tab5:
        st.subheader("Veri kalitesi ve eşleşme kontrolü")

        q1, q2, q3, q4 = st.columns(4)

        q1.metric("SCADA ham kayıt", f"{len(scada_raw):,}")
        q2.metric("SCADA temiz kayıt", f"{len(scada_clean):,}")
        q3.metric("CBS toplam trafo", f"{len(cbs_raw):,}")
        q4.metric("CBS SCADA-RTU = Evet", f"{int(cbs_raw['scada_rtu_var'].sum()):,}")

        st.markdown("#### CBS'de SCADA-RTU = Evet ama SCADA demand eşleşmeyenler")

        if cbs_no_data.empty:
            st.success("Kayıt yok.")
        else:
            st.dataframe(
                cbs_no_data[
                    [
                        "montaj_yeri",
                        "tr_code",
                        "asset_id",
                        "mahalle",
                        "kurulu_guc_kva",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("#### SCADA demand var ama CBS SCADA-RTU = Evet listesinde karşılığı olmayanlar")

        if scada_no_cbs.empty:
            st.success("Kayıt yok.")
        else:
            st.dataframe(scada_no_cbs, use_container_width=True, hide_index=True)

        with st.expander("Eşleşme mantığı"):
            st.write(
                "SCADA `T-4092` → CBS `T4092`; "
                "SCADA `RP_4004 DM` veya `4004 DM` → CBS `B4004`; "
                "SCADA `Enan1/Enan2` → CBS `TR1/TR2`. "
                "H01/H03 gibi alanlar trafo kodu değil, hücre/nokta bilgisidir."
            )


if __name__ == "__main__":
    main()
