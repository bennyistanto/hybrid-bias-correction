---
title: What it fixes
---

<style>
.big { font-size: 1.9rem; font-weight: 700; display: block; line-height: 1.1; }
table.scorecard { width: 100%; border-collapse: collapse; font-size: 13px; }
table.scorecard th, table.scorecard td { padding: 3px 8px; text-align: right; border-bottom: 1px solid var(--theme-foreground-faintest); }
table.scorecard th:first-child, table.scorecard td:first-child,
table.scorecard th:nth-child(2), table.scorecard td:nth-child(2) { text-align: left; }
table.scorecard td.best { background: #e7f5e7; font-weight: 700; }
table.scorecard tr.rule td { border-top: 2px solid var(--theme-foreground-muted); }
</style>

# Staged skill against the gauge

The correction is built in three stages - Linear Scaling (LS), then LS + Empirical Quantile Mapping + a GPD tail (LSEQM), then a CNN refinement (LSEQM+DL). It is scored two ways: **out-of-sample** against the 172 BMKG stations, which are independent of the fitting but not of the reference dataset (CPC-UNI ingests BMKG-derived GTS reports, so the two share some source observations), and **in-sample** against the CPC-UNI target the correction was trained on. Together they show the marginal distribution moving to the gauge while the day-by-day timing does not.

```js
import {fmt, STAGE_KEY, STAGE_COLORS} from "./components/util.js";
const headline = await FileAttachment("data/headline.json").json();
const s = headline.stats;
const STAGES = ["LS", "LSEQM", "LSEQM+DL"];
```

## What each stage adds

Each stage moves the metrics matched to its design dimension and leaves the others alone - which is why the pipeline is built in three steps, not one.

```js
const ATTR = [
  {stage: "LS", add: html`Matches the dekadal mean - it fixes the level, not the shape. SDR <b>0.71</b>, wet-day frequency <b>1.21</b> (over-detects drizzle), upper tail truncated at Q99 <b>0.71</b>.`},
  {stage: "LSEQM", add: html`The largest single jump. EQM plus a GPD tail rescale the whole distribution: SDR <b>0.71 → 1.03</b>, KS p <b>0.01% → 19%</b>, Q99 <b>0.71 → 1.05</b>. The cost is a designed detection trade-off, POD <b>0.78 → 0.65</b>.`},
  {stage: "LSEQM+DL", add: html`Targeted tail refinement. The CNN acts only above the 80th percentile - a design setting, not a formally optimised one (<a href="./sensitivity">sensitivity</a>): Q99 <b>1.05 → 1.01</b>, SDR <b>1.03 → 1.00</b>. The bulk distribution and the detection scores barely move.`}
];
display(html`<div class="grid grid-cols-3">${ATTR.map((a) => html`<div class="finding-card">
  <span class="stage-tag" style=${`background:${STAGE_COLORS[a.stage]}`}>${a.stage}</span>
  <p style="margin:0.5rem 0 0; font-size:13px">${a.add}</p>
</div>`)}</div>`);
```

## Skill by metric

Pick a reference, then a metric: the three stages side by side against the perfect-agreement target. The CPC-UNI reference adds the temporal-skill metrics that only exist in-sample.

```js
const ref = view(Inputs.radio(["BMKG · out-of-sample", "CPC-UNI · in-sample"], {value: "BMKG · out-of-sample", label: "Reference"}));
```

```js
const refRows = ref.startsWith("BMKG") ? headline.bmkg : headline.cpc;
const metric = view(Inputs.select(refRows.map((m) => m.metric), {label: "Metric", value: "SDR"}));
```

```js
const row = refRows.find((m) => m.metric === metric) ?? refRows[0];
const bars = [["LS", row.ls], ["LSEQM", row.lseqm], ["LSEQM+DL", row.lseqmdl]].map(([stage, v]) => ({stage, v}));
const vals = [row.ls, row.lseqm, row.lseqmdl, row.target];
const lo = Math.min(0, ...vals), hi = Math.max(...vals);
const pad = ((hi - lo) || 1) * 0.16;
```

```js
display(Plot.plot({
  width: width,
  height: 320,
  marginLeft: 62,
  x: {domain: STAGES, label: null},
  y: {domain: [lo - (lo < 0 ? pad : 0), hi + pad], grid: true, label: `↑ ${metric}`},
  color: {domain: STAGES, range: STAGES.map((st) => STAGE_COLORS[st])},
  marks: [
    Plot.barY(bars, {x: "stage", y: "v", fill: "stage", tip: true, channels: {stage: "stage", [metric]: "v"}}),
    Plot.text(bars, {x: "stage", y: "v", text: (d) => fmt(d.v), dy: (d) => d.v >= 0 ? -8 : 12, fontSize: 12}),
    Plot.ruleY([row.target], {stroke: "#b2182b", strokeDasharray: "5,4"}),
    Plot.text([{}], {frameAnchor: "top-right", y: row.target, dx: -6, dy: -3, text: [`target ${fmt(row.target)}`], fill: "#b2182b", fontSize: 11, textAnchor: "end"}),
    Plot.ruleY([0])
  ]
}));
```

<div class="note"><b>${row.pillar}</b> &middot; ${metric} &middot; perfect agreement at <b>${fmt(row.target)}</b>. Best stage shaded in the tables below.</div>

```js
function bestStage(m) {
  const v = {ls: m.ls, lseqm: m.lseqm, lseqmdl: m.lseqmdl};
  const score = (x) => m.goal === "high" ? x : m.goal === "low" ? -x : m.goal === "zero" ? -Math.abs(x) : -Math.abs(x - 1);
  return ["ls", "lseqm", "lseqmdl"].reduce((a, b) => score(v[b]) > score(v[a]) ? b : a);
}
function scoretable(rows) {
  return html`<table class="scorecard">
    <thead><tr><th>Pillar</th><th>Metric</th><th>LS</th><th>LSEQM</th><th>LSEQM+DL</th><th>Target</th></tr></thead>
    <tbody>${rows.map((m, i) => {
      const b = bestStage(m);
      const newPillar = i > 0 && rows[i - 1].pillar !== m.pillar;
      return html`<tr class=${newPillar ? "rule" : ""}>
        <td>${m.pillar}</td><td>${m.metric}</td>
        <td class=${b === "ls" ? "best" : ""}>${fmt(m.ls)}</td>
        <td class=${b === "lseqm" ? "best" : ""}>${fmt(m.lseqm)}</td>
        <td class=${b === "lseqmdl" ? "best" : ""}>${fmt(m.lseqmdl)}</td>
        <td>${fmt(m.target)}</td>
      </tr>`;
    })}</tbody>
  </table>`;
}
```

## Out-of-sample: vs 172 BMKG stations

The BMKG stations are held out of the fitting entirely, so this is an out-of-sample test. They are not fully independent of the reference dataset, however: CPC-UNI is a gauge analysis that ingests BMKG-derived GTS reports, so the two share some source observations. Three pillars move to target; event detection is a designed trade-off (unpacked in [detection by threshold](./skill)).

```js
display(scoretable(headline.bmkg));
```

## In-sample: vs CPC-UNI (the calibration target)

The full two-tier picture, including the **Temporal Skill** rows that live only in-sample. LS has zero bias by construction here; LSEQM/LSEQM+DL slightly overshoot CPC-UNI at the upper tail - yet match BMKG almost exactly, because CPC-UNI itself under-catches heavy rain.

```js
display(scoretable(headline.cpc));
```

<div class="note" style="margin-top:1.5rem">
The pattern in one place, and the two tables above are against <i>different</i> references, so read them separately. Against the independent BMKG stations the corrected product lands close to the gauge on the distributional pillars: the standard-deviation ratio reaches <b>1.00</b> and the Q99 ratio <b>1.01</b>. Against CPC-UNI, the dataset the correction was fitted to, the same ratios <i>overshoot</i> to <b>1.15</b> and <b>1.20</b>. Timing does not improve against either: daily Pearson <i>r</i> at the native window holds near <b>${s.r_flat_cpc}</b> against CPC-UNI and <b>${s.r_flat_bmkg}</b> against BMKG, both built the same way - a median across the reference's own units (land pixels for CPC-UNI, stations for BMKG) within each dekad, then averaged over the 36 dekads - and RMSE and NSE, which are computed against CPC-UNI only, do not improve. Why the timing track stays flat is the subject of [the timing ceiling](./ceiling); part of it is a [calendar-window artefact](./window) in the BMKG comparison specifically.
</div>
