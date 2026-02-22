import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="AYEDAŞ | Hamule Ölçüm Zamanı Öneri Paneli", layout="wide")

# =============================
# AYAR: Repo içindeki Excel dosyası
# =============================
DATA_FILE = "Sancaktepe Trafo demand 2025.xlsx"  # repo içine bu isimle koy

# -----------------------------
# Yardımcılar: Excel okuma + temizleme
# -----------------------------
def load_scada_excel(file_path: str) -> pd.DataFrame:
    """
    SCADA export'ta header kaymış olabiliyor.
    header=None okuyup, 'Point Name' satırını bulup onu header yapıyoruz.
    """
    raw = pd.read_excel(file_path, sheet_name=0, header=None)

    header_row_idx = None
    for i in range(min(10, len(raw))):
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
            f"Excel formatında beklenen kolonlar bulunamadı: {missing}. Mevcut kolonlar: {list(df.columns)}"
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    if "quality" not in df.columns:
        df["quality"] = "Unknown"

    df = df.dropna(subset=["point_name", "timestamp", "value"])
    df["point_name"] = df["point_name"].astype(str)

    return df[["point_name", "timestamp", "value", "quality"]]


def extract_dm_and_trafo(point_name: str):
    """
    Yeni gerçeklik:
    - /.../T-4014/...  -> Dağıtım Merkezi (dm_id)
    - ... Enan1 H03 ... -> Trafo ID (trafo_id)
    - /S -> metric

    Örn:
    /Net-E/SANCAKTEPE OM DTM/T-4014/0.4kV/Enan1 H03/S
    dm_id = T-4014
    trafo_id = H03
    metric = S
    """
    dm_id = None
    trafo_id = None
    metric = None

    m = re.search(r"/(T-\d+)", point_name)
    if m:
        dm_id = m.group(1)

    m2 = re.search(r"\b(H\d{2})\b", point_name)
    if m2:
        trafo_id = m2.group(1)

    m3 = re.search(r"/([A-Za-z0-9]+)\s*$", point_name.strip())
    if m3:
        metric = m3.group(1)

    return dm_id, trafo_id, metric


def is_quality_valid(q: str) -> bool:
    """
    Eski hata: contains("valid") -> "invalid" da yakalanıyordu.
    Doğrusu: valid içeriyor ama invalid içermiyor.
    """
    s = str(q).strip().lower()
    return ("valid" in s) and ("invalid" not in s)


def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    parsed = d["point_name"].apply(extract_dm_and_trafo)
    d["dm_id"] = parsed.apply(lambda x: x[0])
    d["trafo_id"] = parsed.apply(lambda x: x[1])
    d["metric"] = parsed.apply(lambda x: x[2])

    d = d.dropna(subset=["dm_id", "trafo_id"])
    d["hour"] = d["timestamp"].dt.floor("H")
    return d


def hourly_aggregate(d: pd.DataFrame) -> pd.DataFrame:
    """
    H'ler ayrı trafo olduğundan SUM/MAX yok.
    Aynı trafo aynı saat içinde birden çok kayıt varsa:
    - Demand için en güvenlisi: MAX (pik yakalama)
    """
    g = d.groupby(["dm_id", "trafo_id", "hour"], as_index=False)["value"].max()
    g = g.rename(columns={"hour": "timestamp", "value": "demand_kva"})
    return g.sort_values(["dm_id", "trafo_id", "timestamp"])


# -----------------------------
# Skorlama: hamule zamanı öner
# -----------------------------
def score_windows(g: pd.DataFrame, window_hours: int = 2) -> pd.DataFrame:
    g = g.copy().sort_values("timestamp")
    x = g["demand_kva"].astype(float)

    roll_mean = x.rolling(window_hours, min_periods=window_hours).mean()
    roll_std = x.rolling(window_hours, min_periods=window_hours).std()
    roll_diff = x.diff().abs().rolling(window_hours, min_periods=window_hours).mean()

    def z(v):
        v = v.to_numpy(dtype=float)
        mu = np.nanmean(v)
        sd = np.nanstd(v) + 1e-9
        return (v - mu) / sd

    mean_z = z(roll_mean)
    std_z = z(roll_std)
    diff_z = z(roll_diff)

    score = (1.2 * mean_z) - (0.8 * std_z) - (0.6 * diff_z)

    out = pd.DataFrame(
        {
            "timestamp": g["timestamp"].values,
            "window_end": g["timestamp"].values,
            "window_start": (g["timestamp"] - pd.Timedelta(hours=window_hours - 1)).values,
            "score": score,
        }
    ).dropna(subset=["score"])

    return out


def attach_demand_at_window_end(recs: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Öneri pencerelerinin bitiş saatindeki demand değerini de ekleyelim.
    """
    if recs.empty:
        return recs

    h = hourly[["dm_id", "trafo_id", "timestamp", "demand_kva"]].copy()
    r = recs.copy()
    r["window_end"] = pd.to_datetime(r["window_end"])
    merged = r.merge(
        h,
        left_on=["dm_id", "trafo_id", "window_end"],
        right_on=["dm_id", "trafo_id", "timestamp"],
        how="left"
    ).drop(columns=["timestamp"])
    return merged


def pick_recommendations(hourly: pd.DataFrame, window_hours: int, top_k: int, min_gap_hours: int) -> pd.DataFrame:
    recs = []
    for (dm, tid), g in hourly.groupby(["dm_id", "trafo_id"]):
        s = score_windows(g, window_hours=window_hours).sort_values("score", ascending=False)

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
            r.insert(0, "dm_id", dm)
            r.insert(1, "trafo_id", tid)
            recs.append(r)

    if not recs:
        return pd.DataFrame(columns=["dm_id", "trafo_id", "window_start", "window_end", "score"])

    out = pd.concat(recs, ignore_index=True)
    out = out.sort_values(["dm_id", "trafo_id", "score"], ascending=[True, True, False])
    out = attach_demand_at_window_end(out, hourly)
    return out


def pick_monthly_recommendations(hourly: pd.DataFrame, window_hours: int, top_k: int, min_gap_hours: int) -> pd.DataFrame:
    h = hourly.copy()
    h["month"] = h["timestamp"].dt.to_period("M").astype(str)

    recs = []
    for (dm, tid, month), g in h.groupby(["dm_id", "trafo_id", "month"]):
        s = score_windows(g, window_hours=window_hours).sort_values("score", ascending=False)

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
            r.insert(0, "dm_id", dm)
            r.insert(1, "trafo_id", tid)
            r.insert(2, "month", month)
            recs.append(r)

    if not recs:
        return pd.DataFrame(columns=["dm_id", "trafo_id", "month", "window_start", "window_end", "score"])

    out = pd.concat(recs, ignore_index=True)
    out = out.sort_values(["dm_id", "trafo_id", "month", "score"], ascending=[True, True, True, False])
    out = attach_demand_at_window_end(out, hourly)
    return out


# -----------------------------
# UI
# -----------------------------
st.title("⚡ AYEDAŞ Hamule Ölçüm Zamanı Öneri Paneli (kVA Demand)")

with st.sidebar:
    st.header("Veri Kaynağı")
    st.caption("Excel dosyası uygulamaya gömülü (kullanıcı yüklemez).")

    st.divider()
    st.header("Filtreler / Ayarlar")

    only_valid = st.toggle("Sadece 'Valid' kaliteyi kullan", value=True)
    remove_zeros = st.toggle("0 değerleri kaldır", value=True)

    window_hours = st.slider("Ölçüm penceresi (saat)", 1, 6, 2)
    top_k = st.slider("Trafo başına öneri sayısı", 1, 10, 3)
    min_gap = st.slider("Öneriler arası min boşluk (saat)", 1, 168, 24)

# Dosya kontrol
if not Path(DATA_FILE).exists():
    st.error(
        f"Excel dosyası bulunamadı: '{DATA_FILE}'.\n"
        f"Lütfen bu dosyayı Streamlit repo içine aynı isimle yükleyin."
    )
    st.stop()

# Veri oku
try:
    df = load_scada_excel(DATA_FILE)
except Exception as e:
    st.error(f"Dosya okunamadı: {e}")
    st.stop()

# Parse
d = to_hourly(df)

# ✅ Valid filtresi düzeltildi (invalid yakalanmıyor)
if only_valid:
    d = d[d["quality"].apply(is_quality_valid)]

if remove_zeros:
    d = d[d["value"] > 0]

# Saatliğe çevir
hourly = hourly_aggregate(d)

# Özet metrikler
col1, col2, col3, col4 = st.columns(4)
col1.metric("Toplam kayıt (temizlenmiş)", f"{len(d):,}")
col2.metric("Dağıtım merkezi sayısı", f"{hourly['dm_id'].nunique():,}")
col3.metric("Saatlik veri noktası", f"{len(hourly):,}")
col4.metric("Tarih aralığı", f"{hourly['timestamp'].min().date()} → {hourly['timestamp'].max().date()}")

st.divider()

# Dağıtım merkezi seçimi
dms = sorted(hourly["dm_id"].unique().tolist())
selected_dm = st.selectbox("Dağıtım merkezi seç", dms)

# Trafo seçimi (artık Hxx = trafo)
trafos = sorted(hourly.loc[hourly["dm_id"] == selected_dm, "trafo_id"].unique().tolist())
selected_trafo = st.selectbox("Trafo seç (Hxx)", trafos)

g = hourly[(hourly["dm_id"] == selected_dm) & (hourly["trafo_id"] == selected_trafo)].copy()

# ✅ Demand değerinin sayısal karşılığı (kartlar)
latest_val = float(g.sort_values("timestamp")["demand_kva"].iloc[-1]) if len(g) else np.nan
max_val = float(g["demand_kva"].max()) if len(g) else np.nan
avg_val = float(g["demand_kva"].mean()) if len(g) else np.nan

m1, m2, m3 = st.columns(3)
m1.metric("Son demand (kVA)", "-" if np.isnan(latest_val) else f"{latest_val:.2f}")
m2.metric("Maks demand (kVA)", "-" if np.isnan(max_val) else f"{max_val:.2f}")
m3.metric("Ortalama demand (kVA)", "-" if np.isnan(avg_val) else f"{avg_val:.2f}")

# Grafik
st.subheader(f"📈 {selected_dm} | {selected_trafo} Saatlik Demand (kVA)")
fig = px.line(g, x="timestamp", y="demand_kva")
st.plotly_chart(fig, use_container_width=True)

# Öneriler (genel)
st.subheader("🎯 Hamule ölçümü için önerilen zaman pencereleri")
recs = pick_recommendations(hourly, window_hours=window_hours, top_k=top_k, min_gap_hours=min_gap)

if recs.empty:
    st.warning("Öneri üretilemedi. Filtreleri gevşetmeyi dene (Valid kapat / 0 kaldırmayı kapat).")
else:
    rec_sel = recs[(recs["dm_id"] == selected_dm) & (recs["trafo_id"] == selected_trafo)].copy()
    if rec_sel.empty:
        st.info("Bu trafo için öneri çıkmadı. (Veri az/dağınık olabilir)")
    else:
        rec_sel["window_start"] = pd.to_datetime(rec_sel["window_start"])
        rec_sel["window_end"] = pd.to_datetime(rec_sel["window_end"])
        rec_sel["score"] = rec_sel["score"].round(3)
        if "demand_kva" in rec_sel.columns:
            rec_sel["demand_kva"] = rec_sel["demand_kva"].round(2)

        st.dataframe(rec_sel[["window_start", "window_end", "demand_kva", "score"]], use_container_width=True)

    st.divider()
    st.subheader("📋 Tüm trafoların öneri listesi")
    out = recs.copy()
    out["window_start"] = pd.to_datetime(out["window_start"])
    out["window_end"] = pd.to_datetime(out["window_end"])
    out["score"] = out["score"].round(3)
    if "demand_kva" in out.columns:
        out["demand_kva"] = out["demand_kva"].round(2)

    st.dataframe(out[["dm_id", "trafo_id", "window_start", "window_end", "demand_kva", "score"]], use_container_width=True)

    csv = out[["dm_id", "trafo_id", "window_start", "window_end", "demand_kva", "score"]].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Önerileri CSV indir", data=csv, file_name="hamule_olcum_onerileri.csv", mime="text/csv")

# Ay ay öneriler
st.divider()
st.subheader("🗓️ Ay ay hamule ölçüm önerileri")

monthly = pick_monthly_recommendations(
    hourly,
    window_hours=window_hours,
    top_k=top_k,
    min_gap_hours=min_gap,
)

if monthly.empty:
    st.info("Ay bazında öneri üretilemedi. Filtreleri gevşetmeyi dene.")
else:
    msel = monthly[(monthly["dm_id"] == selected_dm) & (monthly["trafo_id"] == selected_trafo)].copy()
    msel["window_start"] = pd.to_datetime(msel["window_start"])
    msel["window_end"] = pd.to_datetime(msel["window_end"])
    msel["score"] = msel["score"].round(3)
    if "demand_kva" in msel.columns:
        msel["demand_kva"] = msel["demand_kva"].round(2)

    st.dataframe(msel[["month", "window_start", "window_end", "demand_kva", "score"]], use_container_width=True)

    csvm = monthly[["dm_id", "trafo_id", "month", "window_start", "window_end", "demand_kva", "score"]].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Ay ay önerileri indir (CSV)",
        data=csvm,
        file_name="hamule_onerileri_aylik.csv",
        mime="text/csv",
    )

st.divider()
with st.expander("🔎 Veri örneği (temizlenmiş)"):
    st.dataframe(d.head(50), use_container_width=True)

with st.expander("🧠 Bu skor neye göre?"):
    st.write(
        f"""
        Skor mantığı:
        - Pencere içi ortalama demand yüksekse skor artar.
        - Pencere içi oynaklık (std) yüksekse skor düşer.
        - Ani değişim (abs diff) yüksekse skor düşer.

        Böylece tek bir sivri pik yerine **yüksek ve stabil** saat aralığını hedefler.
        Seçilen pencere süresi: {window_hours} saat.
        """
    )

