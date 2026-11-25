import math
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Dashboard Perencanaan Produksi dan Biaya UMKM", layout="wide")

# Kompatibel dengan Streamlit lama dan baru (use_container_width -> width="stretch")
try:
    _v = tuple(int(x) for x in st.__version__.split(".")[:2])
except Exception:
    _v = (0, 0)
FULL = {"width": "stretch"} if _v >= (1, 50) else {"use_container_width": True}

# ---------- Data contoh (SIMULASI, ganti dengan data usaha sebenarnya) ----------
SALES0 = pd.DataFrame({
    "bulan": [f"M{i}" for i in range(1, 13)],
    "permintaan": [120, 132, 128, 140, 151, 149, 160, 172, 168, 181, 190, 185],
})
BOM0 = pd.DataFrame({
    "bahan": ["Bahan baku utama", "Bahan penolong", "Kemasan"],
    "satuan": ["kg", "kg", "pcs"],
    "per_unit": [0.30, 0.10, 1.0],
    "harga": [15000, 25000, 800],
    "stok_awal": [50.0, 20.0, 100.0],
})
PAR0 = dict(tarif=25000, waktu=20.0, oh_tetap=1500000, oh_var=500, yield_=95,
            markup=40, jam=160, stok_produk=30, stok_aman=20, horizon=3,
            konf=95, layanan=95, lead=1.0, stok_auto=True, uji_k=3)

st.session_state.setdefault("sales", SALES0.copy())
st.session_state.setdefault("bom", BOM0.copy())
st.session_state.setdefault("par", dict(PAR0))
st.session_state.setdefault("cfg", (3, 0.3, 0.5, 0.2))  # n, alpha, alpha Holt, beta Holt
for _k, _val in PAR0.items():
    st.session_state.par.setdefault(_k, _val)


def rp(x):
    return "Rp{:,.0f}".format(x).replace(",", ".")


def pct(new, old):
    if old is None or not np.isfinite(old) or old == 0 or not np.isfinite(new):
        return np.nan
    return round((new / old - 1) * 100, 2)


def kategori_mape(m):
    return "sangat baik" if m < 10 else "baik" if m < 20 else "cukup" if m < 50 else "kurang"


# ---------- Peramalan ----------
def f_ma(y, n):
    f = [np.nan] * n + [y[i - n:i].mean() for i in range(n, len(y))]
    return np.array(f), (lambda h: np.full(h, y[-n:].mean()))


def f_ses(y, a):
    lvl, f = y[0], [np.nan]
    for t in range(1, len(y)):
        f.append(lvl)
        lvl = a * y[t] + (1 - a) * lvl
    return np.array(f), (lambda h: np.full(h, lvl))


def f_holt(y, a, b):
    lvl, tr, f = y[0], y[1] - y[0], [np.nan]
    for t in range(1, len(y)):
        p = lvl + tr
        f.append(p)
        new = a * y[t] + (1 - a) * p
        tr = b * (new - lvl) + (1 - b) * tr
        lvl = new
    return np.array(f), (lambda h: np.array([lvl + (i + 1) * tr for i in range(h)]))


def series():
    d = pd.to_numeric(st.session_state.sales["permintaan"], errors="coerce").dropna()
    return d.values.astype(float)


def run_all(y, n, a, a2, b2):
    out = {}
    metode = {"Rata-rata bergerak": f_ma(y, n), "Pemulusan eksponensial": f_ses(y, a),
              "Holt (tren)": f_holt(y, a2, b2)}
    for nama, (f, fn) in metode.items():
        m = ~np.isnan(f)
        err = y[m] - f[m]          # galat bertanda (aktual - ramalan)
        e = np.abs(err)
        out[nama] = dict(f=f, fn=fn, m=m, err=err, mad=e.mean(),
                         mape=(e / np.where(y[m] == 0, 1, y[m])).mean() * 100)
    best = min(out, key=lambda k: out[k]["mape"])
    return out, best


def forecast(h):
    y = series()
    if len(y) < 4:
        return None
    n, a, a2, b2 = st.session_state.cfg
    out, best = run_all(y, min(n, len(y) - 1), a, a2, b2)
    return out, best, np.maximum(out[best]["fn"](h), 0)


def holdout(y, k, n, a, a2, b2):
    """Validasi data uji: latih pada data awal, uji pada k bulan terakhir."""
    tr, te = y[:-k], y[-k:]
    n = min(n, len(tr) - 1)
    res = {}
    metode = {"Rata-rata bergerak": f_ma(tr, n), "Pemulusan eksponensial": f_ses(tr, a),
              "Holt (tren)": f_holt(tr, a2, b2)}
    for nama, (_, fn) in metode.items():
        e = np.abs(te - np.asarray(fn(k), dtype=float))
        res[nama] = dict(mad=e.mean(), mape=(e / np.where(te == 0, 1, te)).mean() * 100)
    return res


# ---------- Analisis statistik ----------
def konteks():
    """Ringkasan galat metode terbaik. None jika data kurang dari 6 bulan."""
    y = series()
    if len(y) < 6:
        return None
    n, a, a2, b2 = st.session_state.cfg
    out, best = run_all(y, min(n, len(y) - 1), a, a2, b2)
    err = out[best]["err"]
    s = float(np.std(err, ddof=1)) if len(err) > 1 else 0.0
    return dict(y=y, out=out, best=best, err=err, s=s)


def normalitas(err, alpha):
    if len(err) < 3 or np.ptp(err) == 0:
        return None, None, "data tidak cukup"
    w, p = stats.shapiro(err)
    return float(w), float(p), ("normal" if p > alpha else "tidak normal")


def interval(fc, s, konf):
    z = float(stats.norm.ppf(0.5 + konf / 200))
    return np.maximum(fc - z * s, 0), fc + z * s, z


def stok_aman_auto(par):
    k = konteks()
    if k is None:
        return None
    z = float(stats.norm.ppf(par["layanan"] / 100))
    return int(math.ceil(z * k["s"] * math.sqrt(par["lead"])))


def ppar():
    """Parameter efektif: stok pengaman otomatis menggantikan input manual bila aktif."""
    p = dict(st.session_state.par)
    if p["stok_auto"]:
        ss = stok_aman_auto(p)
        if ss is not None:
            p["stok_aman"] = ss
    return p


def pita(fig, x, lo, hi, konf):
    fig.add_scatter(x=x, y=hi, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=x, y=lo, mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(255,127,14,0.25)", name=f"Interval prediksi {konf}%")


# ---------- Perencanaan produksi ----------
def plan(fc, par):
    rows, stok = [], par["stok_produk"]
    yld = par["yield_"] / 100
    kap = par["jam"] * 60 / par["waktu"]
    for i, d in enumerate(fc, 1):
        prod = max(0.0, d + par["stok_aman"] - stok)
        mulai = prod / yld
        rows.append(dict(bulan=f"Bulan +{i}", ramalan=round(d), stok_awal=round(stok),
                         produksi_baik=round(prod), unit_dimulai=round(mulai),
                         kapasitas=round(kap), utilisasi_persen=round(mulai / kap * 100, 1)))
        stok = stok + prod - d
        rows[-1]["stok_akhir"] = round(stok)
    return pd.DataFrame(rows)


# ---------- Biaya (full costing) ----------
def hitung(par, bom, vol, harga=None):
    yld = par["yield_"] / 100
    bahan = (bom["per_unit"] * bom["harga"]).sum()
    tk = par["waktu"] / 60 * par["tarif"]
    mulai = vol / yld
    total_var = mulai * (bahan + tk + par["oh_var"])
    total = total_var + par["oh_tetap"]
    hpp = total / vol
    if harga is None:
        harga = hpp * (1 + par["markup"] / 100)
    vc = total_var / vol
    bep = par["oh_tetap"] / (harga - vc) if harga > vc else np.inf
    return dict(bahan=bahan, tk=tk, hpp=hpp, harga=harga, vc=vc, bep=bep,
                laba=(harga - hpp) * vol, total=total)


def vol_default():
    r = forecast(st.session_state.par["horizon"])
    return max(1, int(round(r[2].mean()))) if r else 100


# ---------- Navigasi ----------
page = st.sidebar.radio("Menu", ["Beranda", "Data", "Peramalan", "Perencanaan Produksi", "Biaya",
                                 "Analisis Statistik"])
st.sidebar.caption("Data bawaan adalah data simulasi. Ganti dengan data usaha di halaman Data.")
par = st.session_state.par
bom = st.session_state.bom.fillna(0)

if page == "Beranda":
    st.title("Dashboard Perencanaan Produksi dan Biaya UMKM")
    st.caption("Alur: data → peramalan → rencana produksi dan bahan → biaya, harga, dan BEP → validasi statistik.")
    res = forecast(par["horizon"])
    if res is None:
        st.warning("Data penjualan minimal 4 bulan. Lengkapi di halaman Data.")
    else:
        out, best, fc = res
        pp = ppar()
        pl = plan(fc, pp)
        vol = vol_default()
        b = hitung(par, bom, vol)
        c = st.columns(5)
        c[0].metric("Ramalan bulan depan", f"{fc[0]:.0f} unit", best, delta_color="off")
        c[1].metric("Produksi bulan +1", f"{pl.loc[0, 'produksi_baik']} unit")
        c[2].metric("HPP per unit", rp(b["hpp"]))
        c[3].metric("BEP", f"{b['bep']:.0f} unit/bulan" if np.isfinite(b["bep"]) else "tidak tercapai")
        c[4].metric("Laba bulanan (perkiraan)", rp(b["laba"]))

        k = konteks()
        lo = hi = None
        if k:
            lo, hi, z = interval(fc, k["s"], par["konf"])
            _, p_norm, ket = normalitas(k["err"], 1 - par["konf"] / 100)
            m = out[best]["mape"]
            c2 = st.columns(3)
            c2[0].metric("MAPE metode terpilih", f"{m:.1f}%", kategori_mape(m), delta_color="off")
            c2[1].metric(f"Interval prediksi {par['konf']}% (bulan +1)", f"{lo[0]:.0f} – {hi[0]:.0f} unit")
            c2[2].metric("Galat ramalan", ket, None if p_norm is None else f"p = {p_norm:.3f}",
                         delta_color="off")
            st.info(f"Metode {best} dipilih dengan MAPE {m:.1f}% (akurasi {kategori_mape(m)}). "
                    f"Permintaan bulan depan diperkirakan {fc[0]:.0f} unit, dengan rentang {lo[0]:.0f} sampai "
                    f"{hi[0]:.0f} unit pada tingkat kepercayaan {par['konf']}%. "
                    f"Stok pengaman produk yang dipakai: {pp['stok_aman']} unit.")
        else:
            st.caption("Lengkapi data minimal 6 bulan untuk menampilkan validasi statistik.")

        y = series()
        xf = list(range(len(y), len(y) + len(fc)))
        fig = go.Figure()
        fig.add_scatter(y=y, mode="lines+markers", name="Aktual")
        fig.add_scatter(x=xf, y=fc, mode="lines+markers", name="Ramalan")
        if lo is not None:
            pita(fig, xf, lo, hi, par["konf"])
        fig.update_layout(title="Permintaan aktual dan ramalan", xaxis_title="Periode", yaxis_title="Unit")
        st.plotly_chart(fig, **FULL)
        if (pl["utilisasi_persen"] > 100).any():
            st.error("Kapasitas tidak cukup pada sebagian bulan. Lihat halaman Perencanaan Produksi.")

elif page == "Data":
    st.header("Data Usaha")
    st.subheader("1. Riwayat penjualan bulanan")
    st.caption("Disarankan minimal 24 bulan agar uji statistik lebih kuat. Minimal 6 bulan untuk analisis statistik.")
    up = st.file_uploader("Unggah CSV (kolom: bulan, permintaan)", type="csv")
    if up is not None:
        d = pd.read_csv(up)
        if {"bulan", "permintaan"}.issubset(d.columns):
            st.session_state.sales = d[["bulan", "permintaan"]]
            st.success("Data penjualan dimuat.")
        else:
            st.error("Kolom harus bernama bulan dan permintaan.")
    st.download_button("Unduh template CSV penjualan", SALES0.to_csv(index=False), "template_penjualan.csv")
    st.session_state.sales = st.data_editor(st.session_state.sales, num_rows="dynamic",
                                            key="ed_sales", **FULL)
    st.subheader("2. Daftar bahan per unit produk (BOM)")
    st.session_state.bom = st.data_editor(st.session_state.bom, num_rows="dynamic",
                                          key="ed_bom", **FULL)
    st.subheader("3. Parameter produksi dan biaya")
    c = st.columns(3)
    par["waktu"] = c[0].number_input("Waktu baku (menit/unit)", 0.1, value=float(par["waktu"]))
    par["tarif"] = c[1].number_input("Tarif tenaga kerja (Rp/jam)", 0, value=int(par["tarif"]))
    par["yield_"] = c[2].number_input("Yield (% produk lolos)", 1, 100, int(par["yield_"]))
    par["oh_tetap"] = c[0].number_input("Overhead tetap (Rp/bulan)", 0, value=int(par["oh_tetap"]))
    par["oh_var"] = c[1].number_input("Overhead variabel (Rp/unit)", 0, value=int(par["oh_var"]))
    par["markup"] = c[2].number_input("Markup harga jual (%)", 0, value=int(par["markup"]))
    par["jam"] = c[0].number_input("Jam kerja tersedia (jam/bulan)", 1, value=int(par["jam"]))
    par["stok_produk"] = c[1].number_input("Stok awal produk jadi (unit)", 0, value=int(par["stok_produk"]))
    par["stok_aman"] = c[2].number_input("Stok pengaman produk manual (unit)", 0, value=int(par["stok_aman"]),
                                         help="Dipakai hanya jika stok pengaman otomatis dimatikan di halaman Analisis Statistik.")
    par["horizon"] = c[0].slider("Horizon perencanaan (bulan)", 1, 6, int(par["horizon"]))

elif page == "Peramalan":
    st.header("Peramalan Permintaan")
    n0, a0, a20, b20 = st.session_state.cfg
    c = st.columns(4)
    n = c[0].slider("Periode rata-rata bergerak", 2, 6, n0)
    a = c[1].slider("Alpha (pemulusan)", 0.05, 0.95, a0, 0.05)
    a2 = c[2].slider("Alpha (Holt)", 0.05, 0.95, a20, 0.05)
    b2 = c[3].slider("Beta (Holt)", 0.05, 0.95, b20, 0.05)
    st.session_state.cfg = (n, a, a2, b2)
    res = forecast(par["horizon"])
    if res is None:
        st.warning("Data penjualan minimal 4 bulan.")
    else:
        out, best, fc = res
        tab = pd.DataFrame([{"Metode": k_, "MAD": round(v["mad"], 2), "MAPE (%)": round(v["mape"], 2),
                             "Akurasi": kategori_mape(v["mape"]),
                             "Ramalan bulan +1": round(float(v["fn"](1)[0]), 1)} for k_, v in out.items()])
        st.dataframe(tab, hide_index=True, **FULL)
        st.success(f"Metode terpilih: {best} (MAPE terkecil). Ramalan {par['horizon']} bulan: "
                   + ", ".join(f"{x:.0f}" for x in fc))
        y = series()
        xf = list(range(len(y), len(y) + len(fc)))
        fig = go.Figure()
        fig.add_scatter(y=y, mode="lines+markers", name="Aktual")
        for k_, v in out.items():
            fig.add_scatter(y=v["f"], mode="lines", name=k_)
        fig.add_scatter(x=xf, y=fc, mode="lines+markers", name=f"Ramalan ({best})")
        k = konteks()
        if k:
            lo, hi, _ = interval(fc, k["s"], par["konf"])
            pita(fig, xf, lo, hi, par["konf"])
            st.caption(f"Pita oranye adalah interval prediksi {par['konf']}%. Uji lengkapnya ada di halaman Analisis Statistik.")
        fig.update_layout(xaxis_title="Periode", yaxis_title="Unit")
        st.plotly_chart(fig, **FULL)

elif page == "Perencanaan Produksi":
    st.header("Perencanaan Produksi dan Kebutuhan Bahan")
    res = forecast(par["horizon"])
    if res is None:
        st.warning("Data penjualan minimal 4 bulan.")
    else:
        out, best, fc = res
        pp = ppar()
        pl = plan(fc, pp)
        auto = par["stok_auto"] and stok_aman_auto(par) is not None
        st.caption(f"Dasar: ramalan metode {best}. Produksi = ramalan + stok pengaman − stok awal, "
                   "unit dimulai = produksi ÷ yield, kapasitas = jam tersedia × 60 ÷ waktu baku. "
                   f"Stok pengaman {pp['stok_aman']} unit "
                   f"({'otomatis dari simpangan galat ramalan' if auto else 'input manual'}).")
        st.dataframe(pl, hide_index=True, **FULL)
        fig = go.Figure()
        fig.add_bar(x=pl["bulan"], y=pl["unit_dimulai"], name="Unit dimulai")
        fig.add_scatter(x=pl["bulan"], y=pl["kapasitas"], name="Kapasitas", mode="lines+markers")
        st.plotly_chart(fig, **FULL)
        if (pl["utilisasi_persen"] > 100).any():
            st.error("Ada bulan dengan utilisasi di atas 100 persen. Tambah jam kerja, tenaga, atau geser produksi.")
        st.subheader("Kebutuhan bahan (MRP sederhana)")
        tab = pd.DataFrame({"bahan": bom["bahan"], "satuan": bom["satuan"]})
        for _, r in pl.iterrows():
            tab[r["bulan"]] = (r["unit_dimulai"] * bom["per_unit"]).round(2)
        kol = list(pl["bulan"])
        tab["total"] = tab[kol].sum(axis=1).round(2)
        tab["stok_awal"] = bom["stok_awal"]
        tab["perlu_dibeli"] = (tab["total"] - tab["stok_awal"]).clip(lower=0).round(2)
        st.dataframe(tab, hide_index=True, **FULL)
        st.download_button("Unduh rencana (CSV)", pl.to_csv(index=False), "rencana_produksi.csv")

elif page == "Biaya":
    st.header("Harga Pokok Produksi, Harga Jual, dan BEP")
    vol = st.number_input("Volume produksi baik per bulan (unit)", 1, value=vol_default())
    b = hitung(par, bom, vol)
    c = st.columns(4)
    c[0].metric("HPP per unit", rp(b["hpp"]))
    c[1].metric("Harga jual", rp(b["harga"]))
    c[2].metric("BEP", f"{b['bep']:.0f} unit" if np.isfinite(b["bep"]) else "tidak tercapai")
    c[3].metric("Laba bulanan", rp(b["laba"]))
    if np.isfinite(b["bep"]):
        aman = (vol - b["bep"]) / vol * 100
        if vol > b["bep"]:
            st.success(f"Volume {vol} unit berada di atas BEP ({b['bep']:.0f} unit). Margin keamanan {aman:.0f}%: "
                       "penjualan boleh turun sebesar itu sebelum usaha mulai rugi.")
        else:
            st.error(f"Volume {vol} unit berada di bawah BEP ({b['bep']:.0f} unit), sehingga usaha merugi.")
    else:
        st.error("Harga jual tidak lebih tinggi dari biaya variabel, BEP tidak tercapai. Naikkan harga atau turunkan biaya.")
    rinci = pd.DataFrame({
        "Komponen": ["Bahan per unit dimulai", "Tenaga kerja per unit", "Overhead variabel per unit",
                     "Overhead tetap per bulan", "Total biaya produksi per bulan"],
        "Nilai": [rp(b["bahan"]), rp(b["tk"]), rp(par["oh_var"]), rp(par["oh_tetap"]), rp(b["total"])]})
    st.dataframe(rinci, hide_index=True, **FULL)

    st.subheader("Grafik titik impas (BEP)")
    if np.isfinite(b["bep"]):
        xs = np.linspace(0, max(vol, b["bep"]) * 1.5, 60)
        fig = go.Figure()
        fig.add_scatter(x=xs, y=b["harga"] * xs, mode="lines", name="Pendapatan")
        fig.add_scatter(x=xs, y=par["oh_tetap"] + b["vc"] * xs, mode="lines", name="Total biaya")
        fig.add_scatter(x=[b["bep"]], y=[b["harga"] * b["bep"]], mode="markers+text",
                        text=[f"BEP {b['bep']:.0f} unit"], textposition="top left",
                        marker=dict(size=11), name="BEP")
        fig.add_vline(x=vol, line_dash="dash", annotation_text="Volume saat ini")
        fig.update_layout(xaxis_title="Volume penjualan (unit/bulan)", yaxis_title="Rupiah per bulan")
        st.plotly_chart(fig, **FULL)

    st.subheader("Analisis sensitivitas (harga jual dianggap tetap)")
    skenario = {
        "Harga bahan +10%": lambda p, bm, v: (p, bm.assign(harga=bm["harga"] * 1.1), v),
        "Waktu baku +10%": lambda p, bm, v: ({**p, "waktu": p["waktu"] * 1.1}, bm, v),
        "Yield −10%": lambda p, bm, v: ({**p, "yield_": p["yield_"] * 0.9}, bm, v),
        "Volume −10%": lambda p, bm, v: (p, bm, v * 0.9),
    }
    rows, dampak = [{"Skenario": "Kondisi dasar", "HPP baru": rp(b["hpp"]), "Δ HPP (%)": 0.0,
                     "BEP (unit)": f"{b['bep']:.0f}" if np.isfinite(b["bep"]) else "tidak tercapai",
                     "Δ BEP (%)": 0.0, "Laba/bulan": rp(b["laba"]), "Δ laba (%)": 0.0}], {}
    for nama, fn in skenario.items():
        p2, bm2, v2 = fn(par, bom, vol)
        h2 = hitung(p2, bm2, v2, harga=b["harga"])
        d_laba = pct(h2["laba"], b["laba"])
        dampak[nama] = abs(d_laba) if np.isfinite(d_laba) else 0
        rows.append({"Skenario": nama, "HPP baru": rp(h2["hpp"]), "Δ HPP (%)": pct(h2["hpp"], b["hpp"]),
                     "BEP (unit)": f"{h2['bep']:.0f}" if np.isfinite(h2["bep"]) else "tidak tercapai",
                     "Δ BEP (%)": pct(h2["bep"], b["bep"]),
                     "Laba/bulan": rp(h2["laba"]), "Δ laba (%)": d_laba})
    st.dataframe(pd.DataFrame(rows), hide_index=True, **FULL)
    if dampak:
        top = max(dampak, key=dampak.get)
        st.info(f"Faktor yang paling memengaruhi laba pada skenario ini: {top}. "
                "Prioritaskan pengendalian faktor tersebut.")

elif page == "Analisis Statistik":
    st.header("Analisis Statistik")
    st.caption("Memeriksa apakah hasil peramalan dapat dipercaya dan apakah perbedaan antarmetode bermakna.")
    k = konteks()
    if k is None:
        st.warning("Analisis statistik membutuhkan data penjualan minimal 6 bulan. Lengkapi di halaman Data.")
        st.stop()
    y, out, best, err, s = k["y"], k["out"], k["best"], k["err"], k["s"]

    KONF, LAYANAN = [90, 95, 99], [85, 90, 95, 97, 99]
    c = st.columns(3)
    par["konf"] = c[0].selectbox("Tingkat kepercayaan (%)", KONF, index=KONF.index(par["konf"]))
    par["layanan"] = c[1].selectbox("Tingkat layanan stok pengaman (%)", LAYANAN, index=LAYANAN.index(par["layanan"]))
    par["lead"] = c[2].number_input("Waktu tunggu (bulan)", 0.25, 6.0, float(par["lead"]), 0.25)
    alpha = 1 - par["konf"] / 100
    st.caption(f"Metode terpilih: {best}. Taraf signifikansi uji α = {alpha:.2f}.")

    # 1. Kenormalan galat
    st.subheader("1. Uji kenormalan galat (Shapiro-Wilk)")
    rows = []
    for nama, v in out.items():
        e = v["err"]
        w, p, ket = normalitas(e, alpha)
        rows.append({"Metode": nama, "n galat": len(e), "Rata-rata galat (bias)": round(float(e.mean()), 2),
                     "Simpangan baku galat": round(float(np.std(e, ddof=1)), 2),
                     "W": np.nan if w is None else round(w, 3),
                     "p-value": np.nan if p is None else round(p, 3), "Kesimpulan": ket})
    st.dataframe(pd.DataFrame(rows), hide_index=True, **FULL)
    _, p_best, ket_best = normalitas(err, alpha)
    if ket_best == "normal":
        st.success(f"Galat metode {best} berdistribusi normal (p = {p_best:.3f} > {alpha:.2f}), "
                   "sehingga interval prediksi dapat dipercaya.")
    elif ket_best == "tidak normal":
        st.warning(f"Galat metode {best} tidak berdistribusi normal (p = {p_best:.3f} ≤ {alpha:.2f}). "
                   "Interval prediksi hanya perkiraan kasar.")
    fig = go.Figure(go.Histogram(x=err, nbinsx=8))
    fig.update_layout(title=f"Sebaran galat metode {best}", xaxis_title="Galat (aktual − ramalan)",
                      yaxis_title="Frekuensi")
    st.plotly_chart(fig, **FULL)

    # 2. Interval prediksi
    st.subheader("2. Interval prediksi ramalan")
    fc = forecast(par["horizon"])[2]
    lo, hi, z = interval(fc, s, par["konf"])
    st.dataframe(pd.DataFrame({"Bulan": [f"Bulan +{i}" for i in range(1, len(fc) + 1)],
                               "Ramalan": fc.round(1), "Batas bawah": lo.round(1), "Batas atas": hi.round(1)}),
                 hide_index=True, **FULL)
    st.latex(r"F_{t+1} \pm z \cdot s")
    st.caption(f"z = {z:.3f} (tingkat kepercayaan {par['konf']}%), s = {s:.2f} (simpangan baku galat). "
               "Lebar interval memakai galat satu langkah; semakin jauh horizon, ketidakpastian sebenarnya lebih besar.")

    # 3. Uji t berpasangan
    st.subheader("3. Uji t berpasangan antarmetode")
    names = list(out)
    common = np.all([out[n_]["m"] for n_ in names], axis=0)
    ae = {n_: np.abs(y[common] - out[n_]["f"][common]) for n_ in names}
    rows, p_pair = [], {}
    for a_, b_ in combinations(names, 2):
        d = ae[a_] - ae[b_]
        if len(d) < 3 or np.ptp(d) == 0:
            t_, p_ = np.nan, np.nan
            ket = "data tidak cukup"
        else:
            t_, p_ = stats.ttest_rel(ae[a_], ae[b_])
            lebih = a_ if d.mean() < 0 else b_
            ket = f"berbeda nyata, {lebih} lebih akurat" if p_ < alpha else "tidak berbeda nyata"
        p_pair[(a_, b_)] = p_
        rows.append({"Pasangan metode": f"{a_} vs {b_}", "Selisih rata-rata |galat|": round(float(d.mean()), 2),
                     "t": np.nan if np.isnan(t_) else round(float(t_), 3),
                     "p-value": np.nan if np.isnan(p_) else round(float(p_), 3), "Kesimpulan": ket})
    st.dataframe(pd.DataFrame(rows), hide_index=True, **FULL)
    st.caption(f"Dibandingkan pada {int(common.sum())} periode yang sama. H0: tidak ada beda rata-rata galat absolut.")
    ps = [p_ for (a_, b_), p_ in p_pair.items() if best in (a_, b_) and not np.isnan(p_)]
    if ps and all(p_ >= alpha for p_ in ps):
        st.info(f"{best} memiliki MAPE terkecil, tetapi selisihnya dengan metode lain belum signifikan secara statistik. "
                "Metode yang lebih sederhana boleh dipakai dengan akurasi yang setara.")
    elif ps:
        st.success(f"Keunggulan {best} didukung perbedaan yang signifikan pada setidaknya satu pembanding.")

    # 4. Uji tren
    st.subheader("4. Uji signifikansi tren permintaan")
    t_idx = np.arange(1, len(y) + 1)
    lr = stats.linregress(t_idx, y)
    c = st.columns(3)
    c[0].metric("Kemiringan tren", f"{lr.slope:.2f} unit/bulan")
    c[1].metric("p-value", f"{lr.pvalue:.4f}")
    c[2].metric("R²", f"{lr.rvalue ** 2:.3f}")
    if lr.pvalue < alpha:
        arah = "naik" if lr.slope > 0 else "turun"
        st.success(f"Tren {arah} signifikan (p < {alpha:.2f}). Metode yang menangkap tren, seperti Holt, layak dipertimbangkan.")
    else:
        st.info(f"Tren tidak signifikan (p ≥ {alpha:.2f}). Permintaan dapat dianggap relatif stabil.")

    # 5. Stok pengaman
    st.subheader("5. Stok pengaman dari simpangan galat")
    zs = float(stats.norm.ppf(par["layanan"] / 100))
    ss = int(math.ceil(zs * s * math.sqrt(par["lead"])))
    st.latex(r"SS = z \cdot s \cdot \sqrt{L}")
    c = st.columns(4)
    c[0].metric("z (tingkat layanan)", f"{zs:.3f}")
    c[1].metric("s (simpangan baku galat)", f"{s:.2f}")
    c[2].metric("L (waktu tunggu)", f"{par['lead']:.2f} bulan")
    c[3].metric("Stok pengaman", f"{ss} unit")
    par["stok_auto"] = st.checkbox("Pakai stok pengaman otomatis di perencanaan produksi", value=par["stok_auto"])
    st.caption(f"Input manual saat ini {par['stok_aman']} unit, hanya dipakai jika kotak di atas tidak dicentang.")

    # 6. Validasi data uji
    st.subheader("6. Validasi dengan data uji")
    kmax = min(6, len(y) - 4)
    if kmax >= 2:
        kk = st.slider("Jumlah bulan terakhir sebagai data uji", 1, kmax, min(max(1, par["uji_k"]), kmax))
        par["uji_k"] = kk
        n_, a_, a2_, b2_ = st.session_state.cfg
        h = holdout(y, kk, n_, a_, a2_, b2_)
        tab = pd.DataFrame([{"Metode": nm, "MAD (data latih)": round(out[nm]["mad"], 2),
                             "MAPE latih (%)": round(out[nm]["mape"], 2),
                             "MAD (data uji)": round(h[nm]["mad"], 2),
                             "MAPE uji (%)": round(h[nm]["mape"], 2)} for nm in out])
        st.dataframe(tab, hide_index=True, **FULL)
        best_uji = min(h, key=lambda nm: h[nm]["mape"])
        if best_uji == best:
            st.success(f"{best} juga terbaik pada data uji, sehingga pemilihannya konsisten.")
        else:
            st.warning(f"Pada data uji, metode terbaik adalah {best_uji}, berbeda dari {best} pada data latih. "
                       "Pertimbangkan menambah data atau menyetel ulang parameter.")
    else:
        st.caption("Data uji membutuhkan minimal 6 bulan data penjualan.")
