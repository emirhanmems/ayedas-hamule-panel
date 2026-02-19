import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="AYEDAŞ | Hamule Ölçüm Zamanı Öneri Paneli", layout="wide")

# -----------------------------
# Yardımcılar: Excel okuma + temizleme
# -----------------------------
def load_scada_excel(file) -> pd.DataFrame:
    """
    Sizin dosyada header kaymış görünüyor.
    Güvenli yöntem: header=None okuyup, 'Point Name' satırını bulup onu header yapıyoruz.
    """
    raw = pd.read_excel(file, sheet_name=0, header=None)

    # Header satırını bul (genelde 0. satır: Point Name, Time stamp, Milliseconds, Value, Source / Quality)
    header_row_idx = None
    for i in range(min(10, len(raw))):
        row = raw.iloc[i].astype(str).str.lower().tolist()
        if any("point name" in c for c in row) and any("time stamp" in c for c in row):
            header_row_idx = i
            break
    if header_row_idx is None:
        # Bulamazsa, 0. satırı header varsayalım (fallback)
        header_row_idx = 0

    header = raw.iloc[header_row_idx].tolist()
    df = raw.iloc[header_row_idx + 1 :].copy()
    df.columns = header

    # Beklenen kolonları normalize et
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

    # Zorunlu alan kontrolü
    needed = {"point_name", "timestamp", "value"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(
            f"Excel formatında beklenen kolonlar bulunamadı: {missing}. Mevcut kolonlar: {list(df.columns)}"
        )

    # Tip dönüşümleri
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Quality yoksa ekleyelim
    if "quality" not in df.columns:
        df["quality"] = "Unknown"

    # Temizle
    df = df.dropna(subset=["point_name", "timestamp", "value"])
    df["point_name"] = df["point_name"].astype(str)

    return df[["point_name", "timestamp", "value", "quality"]]


def extract_trafo_and_feeder(point_name: str):
    """
    /.../T-4014/0.4kV/Enan1 H03/S  -> trafo: T-4014, feeder: H03, metric: S
    """
    trafo = None
    feeder = None
    metric = None

    m = re.search(r"/(T-\d+)", point_name)
    if m:
        trafo = m.group(1)

    m2 = re.search(r"\b(H\d{2})\b", point_name)
    if m2:
        feeder = m2.group(1)

    # en sondaki /S gibi kısmı yakala
    m3 = re.search(r"/([A-Za-z0-9]+)\s*$", point_name.strip())
    if m3:
        metric = m3.group(1)

    return trafo, feeder, metric


def to_hourly(df: pd.DataFrame, agg_mode: str = "sum") -> pd.DataFrame:
    """
    Event bazlı timestamp'i saatliğe çeker.
    Aynı trafoda birden fazla Hxx varsa:
      - sum: trafonun toplam kVA yükü gibi davranır
      - max: en yüksek çıkışı temsil eder (toplam değil)
    """
    d = df.copy()
    parsed = d["point_name"].apply(extract_trafo_and_feeder)
    d["trafo_id"] = parsed.apply(lambda x: x[0])
    d["feeder"] = parsed.apply(lambda x: x[1])
    d["metric"] = parsed.apply(lambda x: x[2])

    d = d.dropna(subset=["trafo_id"])
    d["hour"] = d["timestamp"].dt.floor("H")
    return d


def hourly_aggregate(d: pd.DataFrame, agg_mode: str = "sum") -> pd.DataFrame:
    if agg_mode == "sum":
        g = d.groupby(["trafo_id", "hour"], as_index=False)["value"].sum()
    else:
        g = d.groupby(["trafo_id", "hour"], as_index=False)["value"].max()

    g = g.rename(columns={"hour": "timestamp", "value": "demand_kva"})
    return g.sort_values(["trafo_id", "timestamp"])


# -----------------------------
# Skorlama: hamule zamanı öner
# -----------------------------
def score_windows(g: pd.DataFrame, window_hours: int = 2) -> pd.DataFrame:
    """
    Skor:
      - Yük yüksek olsun (rolling mean)
      - Stabil olsun (rolling std düşük)
      - Ani değişim az olsun (rolling abs diff düşük)
    """
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


def pick_recommendations(hourly: pd.DataFrame, window_hours: int, top_k: int, min_gap_hours: int) -> pd.DataFrame:
    recs = []
    for tid, g in hourly.groupby("trafo_id"):
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
            r.insert(0, "trafo_id", tid)
            recs.append(r)

    if not recs:
        return pd.DataFrame(columns=["trafo_id", "window_start", "window_end", "score", "timestamp"])

    out = pd.concat(recs, ignore_index=True)
    out = out.sort_values(["trafo_id", "score"], ascending=[True, False])
    return out


# ✅ AY AY ÖNERİ (yeni)
def pick_monthly_recommendations(hourly: pd.DataFrame, window_hours: int, top_k: int, min_gap_hours: int) -> pd.DataFrame:
    """
    Her trafo için AY AY top_k pencere öner.
    """
    h = hourly.copy()
    h["month"] = h["timestamp"].dt.to_period("M").astype(str)

    recs = []
    for (tid, month), g in h.groupby(["trafo_id", "month"]):
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
            r.insert(0, "trafo_id", tid)
            r.insert(1, "month", month)
            recs.append(r)

    if not recs:
        return pd.DataFrame(columns=["trafo_id", "month", "window_start", "window_end", "score"])

    out = pd.concat(recs, ignore_index=True)
    out = out.sort_values(["trafo_id", "month", "score"], ascending=[True, True, False])
    return out


# -----------------------------
# UI
# -----------------------------
st.title("⚡ AYEDAŞ Hamule Ölçüm Zamanı Öneri Paneli (kVA Demand)")

with st.sidebar:
    st.header("Yükleme")
    uploaded = st.file_uploader("SCADA Excel dosyasını seç (.xlsx)", type=["xlsx"])

    st.divider()
    st.header("Filtreler / Ayarlar")

    only_valid = st.toggle("Sadece 'Valid' kaliteyi kullan", value=True)
    agg_mode = st.selectbox(
        "Trafo saatlik birleştirme",
        ["sum", "max"],
        index=0,
        help="sum: H01+H02+H03 toplanır (trafo toplamı gibi). max: sadece en yüksek çıkış.",
    )
    remove_zeros = st.toggle("0 değerleri kaldır", value=True)

    window_hours = st.slider("Ölçüm penceresi (saat)", 1, 6, 2)
    top_k = st.slider("Trafo başına öneri sayısı", 1, 10, 3)
    min_gap = st.slider("Öneriler arası min boşluk (saat)", 1, 168, 24)

    st.divider()
    st.caption("Not: Kurulu güç (kVA) gelince % yüklenme ve risk bandı ekleriz.")

if not uploaded:
    st.info("Sol taraftan Excel dosyanı yükle. Panel otomatik analiz edecek.")
    st.stop()

# Veri oku
try:
    df = load_scada_excel(uploaded)
except Exception as e:
    st.error(f"Dosya okunamadı: {e}")
    st.stop()

# Parse
d = to_hourly(df, agg_mode=agg_mode)

if only_valid:
    d = d[d["quality"].astype(str).str.lower().str.contains("valid")]

if remove_zeros:
    d = d[d["value"] > 0]

# Saatliğe çevir
hourly = hourly_aggregate(d, agg_mode=agg_mode)

# Özet metrikler
col1, col2, col3, col4 = st.columns(4)
col1.metric("Toplam kayıt (temizlenmiş)", f"{len(d):,}")
col2.metric("Trafo sayısı", f"{hourly['trafo_id'].nunique():,}")
col3.metric("Saatlik veri noktası", f"{len(hourly):,}")
col4.metric("Tarih aralığı", f"{hourly['timestamp'].min().date()} → {hourly['timestamp'].max().date()}")

st.divider()

# Trafo seçimi
trafos = sorted(hourly["trafo_id"].unique().tolist())
selected = st.selectbox("Trafo seç", trafos)

g = hourly[hourly["trafo_id"] == selected].copy()

# Grafik
st.subheader(f"📈 {selected} Saatlik Demand (kVA)")
fig = px.line(g, x="timestamp", y="demand_kva")
st.plotly_chart(fig, use_container_width=True)

# Öneriler (genel)
st.subheader("🎯 Hamule ölçümü için önerilen zaman pencereleri")
recs = pick_recommendations(hourly, window_hours=window_hours, top_k=top_k, min_gap_hours=min_gap)

if recs.empty:
    st.warning("Öneri üretilemedi. Filtreleri gevşetmeyi dene (Valid kapat / 0 kaldırmayı kapat).")
else:
    rec_sel = recs[recs["trafo_id"] == selected].copy()
    if rec_sel.empty:
        st.info("Bu trafo için öneri çıkmadı. (Veri az/dağınık olabilir)")
    else:
        rec_sel["window_start"] = pd.to_datetime(rec_sel["window_start"])
        rec_sel["window_end"] = pd.to_datetime(rec_sel["window_end"])
        rec_sel["score"] = rec_sel["score"].round(3)
        st.dataframe(rec_sel[["window_start", "window_end", "score"]], use_container_width=True)

    st.divider()
    st.subheader("📋 Tüm trafoların öneri listesi")
    out = recs.copy()
    out["window_start"] = pd.to_datetime(out["window_start"])
    out["window_end"] = pd.to_datetime(out["window_end"])
    out["score"] = out["score"].round(3)

    st.dataframe(out[["trafo_id", "window_start", "window_end", "score"]], use_container_width=True)

    csv = out[["trafo_id", "window_start", "window_end", "score"]].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Önerileri CSV indir", data=csv, file_name="hamule_olcum_onerileri.csv", mime="text/csv")

# ✅ AY AY ÖNERİLER (yeni bölüm)
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
    msel = monthly[monthly["trafo_id"] == selected].copy()
    msel["window_start"] = pd.to_datetime(msel["window_start"])
    msel["window_end"] = pd.to_datetime(msel["window_end"])
    msel["score"] = msel["score"].round(3)

    st.dataframe(msel[["month", "window_start", "window_end", "score"]], use_container_width=True)

    csvm = monthly[["trafo_id", "month", "window_start", "window_end", "score"]].to_csv(index=False).encode("utf-8")
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

