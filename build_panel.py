"""Build a company-year panel from FSS (persistency, planner settlement) and KNIA
(non-life mis-selling, claim denial) disclosure pages saved under data/.

Outputs
  panel.csv        one row per company x calendar year (2020-2026; FSS covers 2021-2025)
  fss_raw_long.csv every FSS row as published, all periods
"""
import re, html, glob, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
AGG = {"생보사평균", "생보사계", "손보사평균", "손보사계", "합계", "업계평균"}


def tables(path):
    s = open(path, encoding="utf-8", errors="ignore").read()
    out = []
    for tb in re.findall(r"<table.*?</table>", s, re.S):
        rows = []
        for r in re.findall(r"<tr.*?</tr>", tb, re.S):
            rows.append([re.sub(r"\s+", "", html.unescape(re.sub(r"<[^>]+>", "", c)))
                         for c in re.findall(r"<t[dh].*?</t[dh]>", r, re.S)])
        out.append(rows)
    return out


def num(x):
    x = (x or "").replace(",", "")
    try:
        return float(x)
    except ValueError:
        return np.nan


# ---------------------------------------------------------------- canonical company ids
LIFE = {
    "한화": ["한화", "한화생명"], "삼성": ["삼성", "삼성생명"], "흥국": ["흥국", "흥국생명"],
    "교보": ["교보", "교보생명"], "신한": ["신한", "신한생명", "신한라이프생명"],
    "iM라이프": ["DGB", "DGB생명", "IM라이프", "iM라이프"], "KDB": ["KDB", "KDB생명"],
    "미래에셋": ["미래에셋", "미래에셋생명"], "KB라이프": ["KB", "KB생명", "KB라이프", "KB라이프생명"],
    "DB": ["DB", "DB생명"], "하나": ["하나", "하나생명"], "IBK연금": ["IBK연금", "IBK연금보험"],
    "농협": ["농협", "농협생명"], "교보라이프플래닛": ["교보라이프", "교보라이프플래닛"],
    "ABL": ["ABL", "ABL생명"], "동양": ["동양", "동양생명"], "메트라이프": ["메트라이프", "메트라이프생명"],
    "처브라이프": ["처브라이프", "처브라이프생명"], "라이나": ["라이나", "라이나생명"],
    "BNP카디프생명": ["BNP카디프", "카디프생명"], "AIA": ["AIA", "AIA생명"],
    "푸본현대": ["푸본현대", "푸본현대생명"], "푸르덴셜": ["푸르덴셜생명"], "오렌지라이프": ["오렌지라이프"],
}
NONLIFE = {
    "메리츠": ["메리츠", "메리츠화재"], "한화손보": ["한화", "한화손보"], "롯데": ["롯데", "롯데손보"],
    "예별(MG)": ["MG", "MG손보", "엠지손보", "예별(舊MG)", "예별손해보험"], "흥국화재": ["흥국", "흥국화재"],
    "삼성화재": ["삼성", "삼성화재"], "현대해상": ["현대", "현대해상"], "KB손보": ["KB", "KB손보"],
    "DB손보": ["DB", "DB손보"], "농협손보": ["농협", "농협손보"], "하나손보": ["하나", "하나손보"],
    "AIG": ["AIG", "AIG손보"], "AXA": ["악사", "악사손보", "AXA손보"],
    "에이스(라이나손보)": ["ACE", "에이스손보", "라이나(ACE)", "라이나손보(에이스손보)"],
    "신한EZ": ["신한EZ", "신한EZ손해보험"], "카디프손보": ["카디프손보"],
    "카카오페이": ["카카오페이손해보험"],
}
LMAP = {a: k for k, v in LIFE.items() for a in v}
NMAP = {a: k for k, v in NONLIFE.items() for a in v}


# ---------------------------------------------------------------- FSS
def fss_rows(path):
    rows = tables(path)[0]
    sec, seen_life_total = None, False
    for r in rows:
        if len(r) == 5:
            sec, r = r[0], r[1:]
        if len(r) != 4 or r[0] in ("회사명", "보험회사명"):
            continue
        name = r[0]
        if sec is None:                     # old layout: life block first, ends at life total row
            s = "손보" if seen_life_total else "생보"
        else:
            s = sec
        if name in ("생보사평균", "생보사계"):
            seen_life_total = True
        yield s, name, num(r[1]), num(r[2]), num(r[3])


recs = []
for f in sorted(glob.glob(f"{D}/fss_*.html")):
    a, b = re.findall(r"fss_(\d{6})_(\d{6})", f)[0]
    for sec, name, settle, p13, p25 in fss_rows(f):
        recs.append(dict(start=a, end=b, sector=sec, name_fss=name, settle=settle, p13=p13, p25=p25))
fss = pd.DataFrame(recs)
fss.to_csv(f"{HERE}/fss_raw_long.csv", index=False)

cal = fss[(fss.start.str[4:] == "01") & (fss.end.str[4:] == "12") & (fss.start.str[:4] == fss.end.str[:4])].copy()
cal["year"] = cal.start.str[:4].astype(int)
cal["company"] = [("" if n in AGG else (LMAP if s == "생보" else NMAP).get(n)) for s, n in zip(cal.sector, cal.name_fss)]
unmapped = cal[cal.company.isna()]
assert unmapped.empty, unmapped[["sector", "name_fss"]].drop_duplicates()
sector_avg = cal[cal.company == ""].copy()
sector_avg["sector"] = sector_avg.sector.map({"생보": "life", "손보": "nonlife"})
sector_avg = sector_avg.groupby(["sector", "year"])[["settle", "p13", "p25"]].first().reset_index()
cal = cal[cal.company != ""].copy()
cal["sector"] = cal.sector.map({"생보": "life", "손보": "nonlife"})
fss_panel = cal[["sector", "company", "year", "settle", "p13", "p25"]]

# ---------------------------------------------------------------- KNIA (non-life)
ms, cl = [], []
for f in sorted(glob.glob(f"{D}/knia04_*.html")):
    y, h = re.findall(r"knia04_(\d{4})([FL])", f)[0]
    t = tables(f)
    assert t[0][0][-1] == "합계"         # last column = all-channel mis-selling rate
    for r in t[1]:
        if not r or r[0] in AGG:
            continue
        ms.append(dict(name=r[0], year=int(y), half=h, ms=num(r[-1])))
for f in sorted(glob.glob(f"{D}/knia07_*.html")):
    y, h = re.findall(r"knia07_(\d{4})([FL])", f)[0]
    t = tables(f)                      # t[0] header, t[1] long-term insurance rows
    assert t[0][0][1].startswith("청구건수")
    for r in t[1]:
        if len(r) < 9 or r[0] in AGG or r[0] == "회사명":
            continue
        cl.append(dict(name=r[0], year=int(y), half=h, claims=num(r[1]), denied=num(r[4]),
                       claim_pols=num(r[6]), post_claim_cxl=num(r[7])))
ms, cl = pd.DataFrame(ms), pd.DataFrame(cl)
for df in (ms, cl):
    df["company"] = df.name.map(NMAP)
    bad = df[df.company.isna()].name.unique()
    assert len(bad) == 0, bad
ms_y = ms.groupby(["company", "year"]).agg(ms=("ms", "mean"), ms_halves=("ms", "count")).reset_index()
cl_y = cl.groupby(["company", "year"])[["claims", "denied", "claim_pols", "post_claim_cxl"]].sum(min_count=1).reset_index()
cl_y["deny_rate"] = 100 * cl_y.denied / cl_y.claims
cl_y["post_claim_cxl_rate"] = 100 * cl_y.post_claim_cxl / cl_y.claim_pols
knia = ms_y.merge(cl_y[["company", "year", "claims", "deny_rate", "post_claim_cxl_rate"]], on=["company", "year"], how="outer")
knia["sector"] = "nonlife"

panel = fss_panel.merge(knia, on=["sector", "company", "year"], how="outer").sort_values(["sector", "company", "year"])
panel["lapse13"] = 100 - panel.p13
panel["lapse25"] = 100 - panel.p25
panel.to_csv(f"{HERE}/panel.csv", index=False)
sector_avg.to_csv(f"{HERE}/fss_sector_avg.csv", index=False)

if __name__ == "__main__":
    print(panel.groupby(["sector", "year"]).agg(n=("company", "size"), settle=("settle", "count"),
                                                p13=("p13", "count"), ms=("ms", "count")))
    print(panel[panel.sector == "nonlife"].pivot(index="company", columns="year", values="ms"))
    print(sector_avg)
