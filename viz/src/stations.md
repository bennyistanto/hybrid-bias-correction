---
title: Stations & seasons
---

<style>
.big { font-size: 1.6rem; font-weight: 700; }
table.card-table { width: 100%; border-collapse: collapse; font-size: 13px; }
table.card-table th, table.card-table td { padding: 3px 8px; text-align: right; border-bottom: 1px solid var(--theme-foreground-faintest); }
table.card-table th:first-child, table.card-table td:first-child { text-align: left; }
table.card-table td.best { background: #e7f5e7; font-weight: 700; }
</style>

# Per-station validation

Every product scored against the independent BMKG gauges, whole-record (the data behind the thesis Taylor diagram). Switch between **individual stations** - one dot per station for a chosen stage - and a **summary** that greys the stations and drops in one pooled dot per product: the raw satellites (IMERG-L, IMERG-F), the calibration target (CPC-UNI), and the three correction stages.

```js
import {fmt, STAGE_KEY, STAGE_COLORS, REGIONS} from "./components/util.js";
const taylor = FileAttachment("data/taylor.json").json();
const stationMetrics = FileAttachment("data/station_metrics.json").json();
const seasonal = FileAttachment("data/seasonal.json").json();
```

```js
const PRODUCT_COLORS = {cpc: "#6a3d9a", imergl: "#e31a1c", imergf: "#ff7f00", ls: "#9aa7b0", lseqm: "#e0913a", lseqmdl: "#2f7d9e"};
function toXY(r, sdr) { return {tx: sdr * r, ty: sdr * Math.sqrt(Math.max(0, 1 - r * r))}; }
function best(vals, goal) {
  const score = (v) => v == null ? -Infinity : goal === "high" ? v : goal === "low" ? -v : goal === "zero" ? -Math.abs(v) : -Math.abs(v - 1);
  const keys = Object.keys(vals);
  return keys.every((k) => vals[k] == null) ? null : keys.reduce((a, b) => score(vals[b]) > score(vals[a]) ? b : a);
}
```

```js
const mode = view(Inputs.radio(["Individual stations", "Summary · all products"], {value: "Individual stations", label: "View"}));
```

```js
const isSummary = mode.startsWith("Summary");
const stage = view(Inputs.radio(["LS", "LSEQM", "LSEQM+DL"], {value: "LSEQM+DL", label: "Stage (individual view)"}));
```

```js
const stageKeyT = STAGE_KEY[stage];
const indivStations = taylor.stations.filter((s) => s.prod[stageKeyT].r != null && s.prod[stageKeyT].sdr != null);
const selected = view(Inputs.select(indivStations, {
  label: "Station (individual view)",
  format: (d) => `${d.name} - ${d.region}`,
  sort: (a, b) => d3.ascending(a.name, b.name)
}));
```

## Taylor diagram

Angle encodes correlation (the rays and outer labels), radius the standard-deviation ratio (the grey arcs, labelled along the diagonal). The dashed arc is the reference SD = 1; the black star (SD 1, r 1) is a perfect match - closer is better.

```js
const maxR = 2.0;
const CORR = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99];
const SD = [0.5, 1.0, 1.5, 2.0];
const arcT = d3.range(0, Math.PI / 2 + 0.001, 0.02);
const stdArcs = SD.flatMap((s) => arcT.map((t) => ({s, x: s * Math.cos(t), y: s * Math.sin(t)})));
const refArc = arcT.map((t) => ({x: Math.cos(t), y: Math.sin(t)}));
const corrRays = CORR.map((r) => ({r, x: maxR * r, y: maxR * Math.sqrt(1 - r * r)}));
const rayLines = corrRays.flatMap((d) => [{r: d.r, x: 0, y: 0}, {r: d.r, x: d.x, y: d.y}]);
const sdLabels = SD.map((s) => ({s, x: s * Math.cos(Math.PI / 4), y: s * Math.sin(Math.PI / 4)}));
```

```js
const indivPts = indivStations.map((s) => { const p = s.prod[stageKeyT]; return {...s, r: p.r, sdr: p.sdr, ...toXY(p.r, p.sdr)}; });
const sel = indivPts.find((d) => d.id === selected.id);
const greyPts = taylor.stations.filter((s) => s.prod.lseqmdl.r != null).map((s) => toXY(s.prod.lseqmdl.r, s.prod.lseqmdl.sdr));
const summaryPts = taylor.products.map((pr) => { const po = taylor.pooled[pr.key]; return {key: pr.key, label: pr.label, r: po.r, sdr: po.sdr, ...toXY(po.r, po.sdr)}; });
```

```js
display(Plot.plot({
  width: Math.min(width, 560),
  height: Math.min(width, 560),
  aspectRatio: 1,
  x: {domain: [0, maxR], label: "standard deviation (normalized) →", ticks: SD, grid: true},
  y: {domain: [0, maxR], label: "standard deviation (normalized) ↑", ticks: SD, grid: true},
  color: isSummary
    ? {domain: taylor.products.map((p) => p.key), range: taylor.products.map((p) => PRODUCT_COLORS[p.key]), legend: false}
    : {legend: true, label: "Region", domain: REGIONS},
  marks: [
    Plot.line(stdArcs, {x: "x", y: "y", z: "s", stroke: "#dddddd", strokeWidth: 0.8}),
    Plot.line(refArc, {x: "x", y: "y", stroke: "#9ecae1", strokeWidth: 1.2, strokeDasharray: "4,3"}),
    Plot.line(rayLines, {x: "x", y: "y", z: "r", stroke: "#eeeeee", strokeWidth: 0.8}),
    Plot.text(corrRays, {x: "x", y: "y", text: (d) => `${d.r}`, dx: 7, dy: -5, fontSize: 10, fill: "#777777"}),
    Plot.text([{x: maxR * Math.cos(0.87), y: maxR * Math.sin(0.87)}], {x: "x", y: "y", text: ["correlation"], fontSize: 11, fill: "#777777", rotate: -38, dx: 4, dy: -8}),
    Plot.text(sdLabels, {x: "x", y: "y", text: (d) => d.s.toFixed(1), fontSize: 9, fill: "#9a9a9a", dx: -7, dy: 9}),
    Plot.dot([{x: 1, y: 0}], {symbol: "star", r: 9, fill: "#111111"}),
    ...(isSummary ? [
      Plot.dot(greyPts, {x: "tx", y: "ty", r: 2.5, fill: "#dadada"}),
      Plot.dot(summaryPts, {x: "tx", y: "ty", fill: "key", r: 8, stroke: "white", strokeWidth: 1,
        tip: true, channels: {Product: "label", r: "r", "std ratio": "sdr"}}),
      Plot.text(summaryPts, {x: "tx", y: "ty", text: "label", fill: "key", dy: -13, fontSize: 10, fontWeight: 600})
    ] : [
      Plot.dot(indivPts, {x: "tx", y: "ty", fill: "region", r: 3.5, stroke: "white", strokeWidth: 0.4,
        tip: true, channels: {Station: "name", r: "r", "std ratio": "sdr"}}),
      sel ? Plot.dot([sel], {x: "tx", y: "ty", r: 7, fill: "none", stroke: "#111111", strokeWidth: 2}) : null
    ])
  ].filter(Boolean)
}));
```

```js
{
  const box = html`<div></div>`;
  if (isSummary) {
    const P = taylor.products;
    const pooled = taylor.pooled;
    const ROWS = [
      {label: "Pearson r", key: "r", goal: "high"},
      {label: "Std-dev ratio", key: "sdr", goal: "one"},
      {label: "RMSE (mm/day)", key: "rmse", goal: "low"},
      {label: "Mean bias (mm/day)", key: "bias", goal: "zero"}
    ];
    box.append(html`<h2>Pooled over all stations</h2>`);
    box.append(html`<p>One dot per product on the diagram, pooled across the BMKG network. Best per row shaded.</p>`);
    box.append(html`<table class="card-table">
      <thead><tr><th>Metric</th>${P.map((p) => html`<th style=${`color:${PRODUCT_COLORS[p.key]}`}>${p.label}</th>`)}</tr></thead>
      <tbody>${ROWS.map((m) => {
        const vals = Object.fromEntries(P.map((p) => [p.key, pooled[p.key][m.key]]));
        const bk = best(vals, m.goal);
        return html`<tr><td>${m.label}</td>${P.map((p) => html`<td class=${bk === p.key ? "best" : ""}>${fmt(pooled[p.key][m.key])}</td>`)}</tr>`;
      })}</tbody>
    </table>`);
    box.append(html`<div class="keyfinding" style="margin-top:1.2rem">
      <span class="kf-label">Finding</span>
      Every product correlates only <b>${pooled.cpc.r}</b> to <b>${pooled.imergf.r}</b> with the gauges - the timing ceiling holds for the raw satellites and the corrected stages alike. What the correction fixes is spread: LSEQM+DL lands at SDR <b>${pooled.lseqmdl.sdr}</b>, closer to the gauge than even CPC-UNI (<b>${pooled.cpc.sdr}</b>), which under-catches variance.
    </div>`);
  } else {
    const info = selected;
    const det = {ls: stationMetrics.ls.find((d) => d.id === info.id), lseqm: stationMetrics.lseqm.find((d) => d.id === info.id), lseqmdl: stationMetrics.lseqmdl.find((d) => d.id === info.id)};
    const ROWS = [
      {label: "Pearson r", get: (k) => info.prod[k].r, goal: "high"},
      {label: "Std-dev ratio", get: (k) => info.prod[k].sdr, goal: "one"},
      {label: "RMSE (mm/day)", get: (k) => info.prod[k].rmse, goal: "low"},
      {label: "POD", get: (k) => det[k] && det[k].pod, goal: "high"},
      {label: "CSI", get: (k) => det[k] && det[k].csi, goal: "high"},
      {label: "FAR", get: (k) => det[k] && det[k].far, goal: "low"},
      {label: "KS p-value", get: (k) => det[k] && det[k].ks, goal: "high"}
    ];
    box.append(html`<h2>${info.name}</h2>`);
    box.append(html`<p>${info.region} &middot; WMO ${info.id} &middot; ${info.lat}&deg;, ${info.lon}&deg;E &middot; correction stages, best per row shaded.</p>`);
    box.append(html`<table class="card-table">
      <thead><tr><th>Metric</th><th>LS</th><th>LSEQM</th><th>LSEQM+DL</th></tr></thead>
      <tbody>${ROWS.map((m) => {
        const vals = {ls: m.get("ls"), lseqm: m.get("lseqm"), lseqmdl: m.get("lseqmdl")};
        const bk = best(vals, m.goal);
        return html`<tr><td>${m.label}</td>
          <td class=${bk === "ls" ? "best" : ""}>${fmt(vals.ls)}</td>
          <td class=${bk === "lseqm" ? "best" : ""}>${fmt(vals.lseqm)}</td>
          <td class=${bk === "lseqmdl" ? "best" : ""}>${fmt(vals.lseqmdl)}</td>
        </tr>`;
      })}</tbody>
    </table>`);
  }
  display(box);
}
```

<div class="note" style="margin-top: 1.5rem">
Correlation and standard-deviation ratio are whole-record statistics against BMKG (the Taylor source); the detection scores (POD, CSI, FAR, KS) in the individual card are the dekad-pooled medians.
</div>

## Seasonal stability

Dekadal pooling hides the within-year cycle. Re-aggregated to calendar months - the per-pixel spatial median in-sample against CPC-UNI - the four representative metrics show the correction holds across Indonesia's wet and dry seasons.

```js
const SEASON_META = {
  sdr: {label: "Std-dev ratio", ref: 1.0},
  rmse: {label: "RMSE (mm/day)", ref: null},
  ks: {label: "KS p-value (%)", ref: 5},
  csi: {label: "CSI", ref: null}
};
const seasonMetric = view(Inputs.radio(
  new Map([["Std-dev ratio", "sdr"], ["RMSE", "rmse"], ["KS p-value", "ks"], ["CSI", "csi"]]),
  {value: "sdr", label: "Metric"}));
```

```js
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const seasonLong = ["LS", "LSEQM", "LSEQM+DL"].flatMap((s) =>
  seasonal[STAGE_KEY[s]].map((m) => ({stage: s, mname: MONTHS[m.month - 1], v: m[seasonMetric]})));
const seasonRef = SEASON_META[seasonMetric].ref;
```

```js
display(Plot.plot({
  width: width,
  height: 320,
  marginBottom: 40,
  x: {domain: MONTHS, label: null},
  y: {grid: true, label: `↑ ${SEASON_META[seasonMetric].label}`},
  color: {legend: true, domain: ["LS", "LSEQM", "LSEQM+DL"], range: ["LS", "LSEQM", "LSEQM+DL"].map((s) => STAGE_COLORS[s])},
  marks: [
    seasonRef != null ? Plot.ruleY([seasonRef], {stroke: "#999999", strokeDasharray: "4,3"}) : null,
    Plot.line(seasonLong, {x: "mname", y: "v", z: "stage", stroke: "stage", strokeWidth: 2, marker: "circle",
      tip: true, channels: {stage: "stage", month: "mname", value: "v"}})
  ].filter(Boolean)
}));
```

```js
const sdrVals = seasonal.lseqmdl.map((m) => m.sdr);
```

<div class="keyfinding">
<span class="kf-label">Finding</span>
Every metric is flat across the seasonal cycle: LSEQM+DL's SDR holds within <b>${Math.min(...sdrVals).toFixed(2)}</b> to <b>${Math.max(...sdrVals).toFixed(2)}</b> all year and the stage ordering never flips. RMSE tracks rainfall magnitude - peak in the <b>DJF</b> wet season, trough in <b>August</b> - not method skill. The pooled headline is not driven by a handful of dekads.
</div>

<div class="note" style="margin-top: 1rem">
Monthly values are the per-pixel spatial median in-sample against CPC-UNI, averaging the three dekads of each month. The dashed line is the target (SDR = 1) or the 5% KS threshold. Note the KS panel: the corrected stages sit at the test floor against CPC in-sample, the flip side of their strong KS pass against the independent BMKG gauges (out-of-sample vs in-sample). LS reads higher here only because it changes the distribution least.
</div>
