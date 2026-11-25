import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Dashboard Perencanaan Produksi dan Biaya UMKM", layout="wide")

# ---------- Data contoh (SIMULASI) ----------
SALES0 = pd.DataFrame({
    "bulan": [f"M{i}" for i in range(1, 25)],
    "permintaan": [120, 132, 128, 140, 151, 149, 160, 172, 168, 181, 190, 185,
                   198, 205, 201, 214, 223, 220, 231, 243, 238, 250, 259, 255],
})
BOM0 = pd.DataFrame({
    "bahan": ["Bahan baku utama", "Bahan penolong", "Kemasan"],
    "satuan": ["kg", "kg", "pcs"],
    "per_unit": [0.30, 0.10, 1.0],
    "harga": [15000, 25000, 800],
    "stok_awal": [50.0, 20.0, 100.0],
})
METODE = ["Rata-rata bergerak", "Pemulusan eksponensial", "Holt (tren)"]


def rp(x):
    return "Rp{:,.0f}".format(x).replace(",", ".")


def fp(p):
    return "< 0,001" if p < 0.001 else f"{p:.3f}"


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
        f.append(p if t >= 2 else np.nan)  # galat bulan ke-2 selalu nol (inisialisasi), tidak dihitung
        new = a * y[t] + (1 - a) * p
        tr = b * (new - lvl) + (1 - b) * tr
        lvl = new
    return np.array(f), (lambda h: np.array([lvl + (i + 1) * tr for i in range(h)]))


def fit(nama, y, cfg):
    n, a, a2, b2 = cfg
    if nama == METODE[0]:
        return f_ma(y, max(1, min(n, len(y) - 1)))
    if nama == METODE[1]:
        return f_ses(y, a)
    return f_holt(y, a2, b2)


def analisis(y, cfg, n_test, h):
    latih, uji = y[:-n_test], y[-n_test:]
    out = {}
    for nm in METODE:
        _, fn = fit(nm, latih, cfg)
        err = uji - np.maximum(fn(n_test), 0)
        full, fnf = fit(nm, y, cfg)
        out[nm] = dict(mad=np.abs(err).mean(),
                       mape=(np.abs(err) / np.where(uji == 0, 1, uji)).mean() * 100,
                       fitted=full, next1=float(fnf(1)[0]))
    best = min(out, key=lambda k: out[k]["mape"])
    fitted, fn = fit(best, y, cfg)
    resid = (y - fitted)[~np.isnan(fitted)]
    return out, best, np.maximum(fn(h), 0), fitted, resid


# ---------- Produksi dan biaya ----------
def plan(fc, par, ss):
    rows, stok = [], par["stok_produk"]
    yld, kap = par["yield_"] / 100, par["jam"] * 60 / par["waktu"]
    for i, d in enumerate(fc, 1):
        prod = max(0.0, d + ss - stok)
        mulai = prod / yld
        stok = stok + prod - d
        rows.append(dict(bulan=f"Bulan +{i}", ramalan=round(d), stok_pengaman=round(ss),
                         produksi_baik=round(prod), unit_dimulai=round(mulai), kapasitas=round(kap),
                         utilisasi_persen=round(mulai / kap * 100, 1), stok_akhir=round(stok)))
    return pd.DataFrame(rows)


def hitung(par, bom, vol, harga=None):
    yld = par["yield_"] / 100
    bahan = (bom["per_unit"] * bom["harga"]).sum()
    tk = par["waktu"] / 60 * par["tarif"]
    total_var = vol / yld * (bahan + tk + par["oh_var"])
    total = total_var + par["oh_tetap"]
    hpp = total / vol
    harga = harga if harga else hpp * (1 + par["markup"] / 100)
    vc = total_var / vol
    bep = par["oh_tetap"] / (harga - vc) if harga > vc else np.inf
    return dict(bahan=bahan, tk=tk, hpp=hpp, harga=harga, bep=bep, laba=(harga - hpp) * vol, total=total)


def hitung_semua(y, cfg, n_test, h, par, bom, layanan, L):
    out, best, fc, fitted, resid = analisis(y, cfg, n_test, h)
    n, s = len(resid), float(resid.std(ddof=1))
    w, pw = stats.shapiro(resid)
    tk = stats.t.ppf(0.975, n - 1)
    ip = pd.DataFrame({"bulan": [f"Bulan +{i}" for i in range(1, h + 1)], "ramalan": fc.round(0),
                       "batas_bawah": np.maximum(fc - tk * s, 0).round(0), "batas_atas": (fc + tk * s).round(0)})
    lr = stats.linregress(np.arange(len(y)), y)
    z = float(stats.norm.ppf(layanan / 100))
    ss = z * s * np.sqrt(L)
    pl = plan(fc, par, ss)
    mat = pd.DataFrame({"bahan": bom["bahan"], "satuan": bom["satuan"]})
    for _, r in pl.iterrows():
        mat[r["bulan"]] = (r["unit_dimulai"] * bom["per_unit"]).round(2)
    mat["total"] = mat[list(pl["bulan"])].sum(axis=1).round(2)
    mat["stok_awal"] = bom["stok_awal"]
    mat["perlu_dibeli"] = (mat["total"] - mat["stok_awal"]).clip(lower=0).round(2)
    vol = max(1, int(round(fc.mean())))
    b = hitung(par, bom, vol)
    kom = pd.DataFrame({"Komponen": ["Bahan per unit dimulai", "Tenaga kerja per unit", "Overhead variabel per unit",
                                     "Overhead tetap per bulan", "Total biaya produksi per bulan"],
                        "Nilai": [rp(b["bahan"]), rp(b["tk"]), rp(par["oh_var"]), rp(par["oh_tetap"]), rp(b["total"])]})
    skenario = {"Harga bahan +10%": lambda: (par, bom.assign(harga=bom["harga"] * 1.1), vol),
                "Waktu baku +10%": lambda: ({**par, "waktu": par["waktu"] * 1.1}, bom, vol),
                "Yield −10%": lambda: ({**par, "yield_": par["yield_"] * 0.9}, bom, vol),
                "Volume −10%": lambda: (par, bom, vol * 0.9)}
    rows = [{"Skenario": "Kondisi dasar", "HPP baru": rp(b["hpp"]), "Perubahan HPP (%)": 0.0, "BEP (unit)": round(b["bep"])}]
    for nama, fn in skenario.items():
        p2, bm2, v2 = fn()
        r2 = hitung(p2, bm2, v2, harga=b["harga"])
        rows.append({"Skenario": nama, "HPP baru": rp(r2["hpp"]),
                     "Perubahan HPP (%)": round((r2["hpp"] / b["hpp"] - 1) * 100, 2), "BEP (unit)": round(r2["bep"])})
    tab_metode = pd.DataFrame([{"Metode": k, "MAD (data uji)": round(v["mad"], 2), "MAPE data uji (%)": round(v["mape"], 2),
                                "Ramalan bulan +1": round(v["next1"], 1)} for k, v in out.items()])
    return dict(y=y, out=out, best=best, fc=fc, fitted=fitted, n=n, s=s, bias=float(resid.mean()), w=w, pw=pw, ip=ip,
                lr=lr, z=z, ss=ss, L=L, pl=pl, mat=mat, vol=vol, b=b, kom=kom, sens=pd.DataFrame(rows),
                tab_metode=tab_metode, n_test=n_test)


def ke_excel(R):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for nama, df in [("Peramalan", R["tab_metode"]), ("Interval Prediksi", R["ip"]), ("Rencana Produksi", R["pl"]),
                         ("Kebutuhan Bahan", R["mat"]), ("Biaya", R["kom"]), ("Sensitivitas", R["sens"])]:
            df.to_excel(w, sheet_name=nama, index=False)
    return buf.getvalue()


def templat_xlsx():
    """Templat data penjualan (.xlsx) dibuat di memori; kolom langsung terpisah rapi di Excel."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        SALES0.to_excel(w, sheet_name="Data Penjualan", index=False)
        ws = w.sheets["Data Penjualan"]
        ws.column_dimensions["A"].width = 12
        ws.column_dimensions["B"].width = 14
    return buf.getvalue()


# ---------- Pengelolaan state ----------
def reset_hasil():
    """Hapus seluruh hasil proses sehingga dashboard kembali ke tampilan awal."""
    st.session_state.R = None
    st.session_state.kosong = 0
    st.session_state.sudah_run = False


for k, v in {"R": None, "kosong": 0, "sudah_run": False, "file_sig": None}.items():
    st.session_state.setdefault(k, v)

# ---------- Panel kiri ----------
sb = st.sidebar
sb.header("Pengaturan")
up = sb.file_uploader("Unggah data penjualan (.csv / .xlsx)", type=["csv", "xlsx"])

# Deteksi file baru / file dihapus lewat identitas file (nama, ukuran, id unggahan)
sig = None if up is None else f"{up.name}|{up.size}|{getattr(up, 'file_id', '')}"
if sig != st.session_state.file_sig:
    reset_hasil()
    st.session_state.file_sig = sig

if up is not None:
    d = pd.read_excel(up) if up.name.lower().endswith("xlsx") else pd.read_csv(up)
    data = d.iloc[:, :2].copy()
    data.columns = ["bulan", "permintaan"]
else:
    data = SALES0
    sb.caption("Belum ada file. Memakai data contoh (simulasi 24 bulan).")
st.session_state.data_aktif = data

sb.subheader("Peramalan")
n_test = sb.slider("Data uji (bulan terakhir)", 2, 6, 3)
h = sb.number_input("Jumlah periode ke depan (bulan)", 1, 6, 3)
with sb.expander("Parameter metode"):
    n_ma = st.slider("Periode rata-rata bergerak", 2, 6, 3)
    a1 = st.slider("Alpha (pemulusan)", 0.05, 0.95, 0.30, 0.05)
    a2 = st.slider("Alpha (Holt)", 0.05, 0.95, 0.50, 0.05)
    b2 = st.slider("Beta (Holt)", 0.05, 0.95, 0.20, 0.05)
sb.subheader("Stok pengaman")
layanan = sb.selectbox("Tingkat layanan (%)", [90, 95, 99], index=1)
L = sb.number_input("Waktu tunggu bahan (bulan)", 0.25, 6.0, 1.0, 0.25)
with sb.expander("Parameter produksi dan biaya"):
    par = dict(
        waktu=st.number_input("Waktu baku (menit/unit)", 0.1, value=30.0),
        tarif=st.number_input("Tarif tenaga kerja (Rp/jam)", 0, value=25000),
        yield_=st.number_input("Yield (% produk lolos)", 1, 100, 95),
        oh_tetap=st.number_input("Overhead tetap (Rp/bulan)", 0, value=1500000),
        oh_var=st.number_input("Overhead variabel (Rp/unit)", 0, value=500),
        markup=st.number_input("Markup harga jual (%)", 0, value=40),
        jam=st.number_input("Jam kerja tersedia (jam/bulan)", 1, value=160),
        stok_produk=st.number_input("Stok awal produk jadi (unit)", 0, value=30))
with sb.expander("Daftar bahan per unit produk (BOM)"):
    bom = st.data_editor(BOM0, num_rows="dynamic", key="bom").fillna(0)

if sb.button("Run Process", type="primary"):
    s = pd.to_numeric(data["permintaan"], errors="coerce")
    kosong = int(s.isna().sum() + (s < 0).sum())
    y = s.where(s >= 0).interpolate(limit_direction="both").dropna().values.astype(float)
    if len(y) < n_test + 6:
        reset_hasil()
        sb.error(f"Data terlalu sedikit. Butuh minimal {n_test + 6} bulan untuk {n_test} bulan data uji.")
    else:
        st.session_state.kosong = kosong
        st.session_state.R = hitung_semua(y, (n_ma, a1, a2, b2), n_test, int(h), par, bom, layanan, L)
        st.session_state.sudah_run = True

sb.button("Kembali ke Tampilan Awal", on_click=reset_hasil,
          disabled=st.session_state.R is None, key="reset_sidebar")

# ---------- Halaman utama ----------
st.title("Dashboard Perencanaan Produksi dan Biaya UMKM")
st.caption("Sistem pendukung keputusan: data penjualan → peramalan → uji statistik → rencana produksi → biaya dan BEP.")
R = st.session_state.get("R")

if R is None:
    st.info("Petunjuk: unggah data di panel kiri (atau pakai data contoh), atur parameter, lalu tekan Run Process.")
    c = st.columns(4)
    for col, (jd, tx) in zip(c, [("1. Unggah data", "File CSV atau Excel berisi kolom bulan dan permintaan."),
                                 ("2. Atur parameter", "Data uji, parameter metode, stok pengaman, produksi, dan biaya."),
                                 ("3. Run Process", "Sistem meramal, menguji, merencanakan produksi, dan menghitung biaya."),
                                 ("4. Unduh hasil", "Seluruh hasil dalam satu file Excel.")]):
        col.markdown(f"**{jd}**\n\n{tx}")
    st.download_button("Unduh templat data (XLSX)", templat_xlsx(), "templat_data.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.caption("Data aktif yang akan diproses:" + (f" {up.name}" if up is not None else " data contoh (simulasi)"))
    st.dataframe(st.session_state.data_aktif, hide_index=True, height=250)
else:
    top = st.columns([1, 1, 4])
    top[0].download_button("Unduh hasil (Excel)", ke_excel(R), "hasil_perencanaan.xlsx")
    top[1].button("Kembali ke Tampilan Awal", on_click=reset_hasil, key="reset_main")
    if st.session_state.get("kosong"):
        st.warning(f"{st.session_state.kosong} nilai kosong atau negatif diisi dengan interpolasi linear.")
    t1, t2, t3, t4 = st.tabs(["Peramalan", "Analisis Statistik", "Perencanaan Produksi", "Biaya"])

    with t1:
        st.dataframe(R["tab_metode"], hide_index=True, use_container_width=True)
        st.success(f"Metode terpilih: {R['best']} (MAPE data uji terkecil pada {R['n_test']} bulan terakhir). "
                   "Ramalan: " + ", ".join(f"{x:.0f}" for x in R["fc"]) + " unit.")
        fig = go.Figure()
        fig.add_scatter(y=R["y"], mode="lines+markers", name="Aktual")
        for k, v in R["out"].items():
            fig.add_scatter(y=v["fitted"], mode="lines", name=k)
        fig.add_scatter(x=list(range(len(R["y"]), len(R["y"]) + len(R["fc"]))), y=R["fc"],
                        mode="lines+markers", name="Ramalan")
        fig.update_layout(xaxis_title="Periode", yaxis_title="Unit")
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.caption(f"Galat peramalan satu langkah metode {R['best']}: n = {R['n']}, simpangan baku s = {R['s']:.2f}, "
                   f"rata-rata galat (bias) = {R['bias']:.2f}. Taraf signifikansi α = 0,05. Sampel kecil, hasil bersifat indikatif.")
        st.subheader("1. Uji kenormalan galat (Shapiro-Wilk)")
        st.write(f"W = {R['w']:.3f}, p-value = {fp(R['pw'])}. "
                 + ("Normalitas tidak ditolak." if R["pw"] > 0.05 else "Normalitas ditolak, interpretasi interval dan stok pengaman perlu hati-hati."))
        st.subheader("2. Interval prediksi 95 persen")
        st.dataframe(R["ip"], hide_index=True, use_container_width=True)
        st.caption("Ramalan ± t(0,975; n−1) × s, memakai galat satu langkah.")
        st.subheader("3. Uji tren permintaan")
        lr = R["lr"]
        st.write(f"Kemiringan = {lr.slope:.2f} unit/bulan, R² = {lr.rvalue ** 2:.3f}, p-value = {fp(lr.pvalue)}. "
                 + ("Tren signifikan, metode yang menangkap tren seperti Holt beralasan dipakai." if lr.pvalue < 0.05 else "Tren tidak signifikan."))
        st.subheader("4. Stok pengaman")
        c = st.columns(4)
        c[0].metric("z (tingkat layanan)", f"{R['z']:.3f}")
        c[1].metric("s (simpangan galat)", f"{R['s']:.2f}")
        c[2].metric("Waktu tunggu (bulan)", f"{R['L']:.2f}")
        c[3].metric("Stok pengaman", f"{R['ss']:.0f} unit")
        st.caption("SS = z × s × √L. Nilai ini dipakai pada rencana produksi.")

    with t3:
        pl = R["pl"]
        st.caption("Produksi = ramalan + stok pengaman − stok awal; unit dimulai = produksi ÷ yield; "
                   "kapasitas = jam tersedia × 60 ÷ waktu baku.")
        st.dataframe(pl, hide_index=True, use_container_width=True)
        fig = go.Figure()
        fig.add_bar(x=pl["bulan"], y=pl["unit_dimulai"], name="Unit dimulai")
        fig.add_scatter(x=pl["bulan"], y=pl["kapasitas"], name="Kapasitas", mode="lines+markers")
        st.plotly_chart(fig, use_container_width=True)
        if (pl["utilisasi_persen"] > 100).any():
            st.error("Utilisasi di atas 100 persen pada sebagian bulan. Tambah jam kerja atau tenaga, atau geser produksi.")
        elif (pl["utilisasi_persen"] >= 90).any():
            st.warning("Utilisasi mendekati batas kapasitas (90 persen atau lebih) pada sebagian bulan.")
        st.subheader("Kebutuhan bahan (MRP sederhana)")
        st.dataframe(R["mat"], hide_index=True, use_container_width=True)

    with t4:
        b = R["b"]
        st.caption(f"Volume produksi baik = rata-rata ramalan ({R['vol']} unit per bulan). Metode full costing.")
        c = st.columns(4)
        c[0].metric("HPP per unit", rp(b["hpp"]))
        c[1].metric("Harga jual", rp(b["harga"]))
        c[2].metric("BEP", f"{b['bep']:.0f} unit" if np.isfinite(b["bep"]) else "tidak tercapai")
        c[3].metric("Laba bulanan*", rp(b["laba"]))
        st.caption("*Dengan asumsi seluruh produk terjual.")
        st.dataframe(R["kom"], hide_index=True, use_container_width=True)
        st.subheader("Analisis sensitivitas (harga jual dianggap tetap)")
        st.dataframe(R["sens"], hide_index=True, use_container_width=True)

st.divider()
st.caption("Dikembangkan untuk Capstone Design Project, Teknik Industri, Universitas Gunadarma. Data contoh adalah simulasi.")
