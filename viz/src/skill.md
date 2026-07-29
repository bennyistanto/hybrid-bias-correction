---
title: Detection by threshold
---

<style>
.callout { border-left: 3px solid var(--theme-foreground-focus); padding: 0.4rem 0 0.4rem 1rem; margin: 1rem 0; }
.callout .big { font-size: 1.5rem; font-weight: 700; }
.legendrow { display: flex; gap: 14px; font-size: 13px; flex-wrap: wrap; }
.dotlab { display: inline-flex; align-items: center; gap: 5px; }
.dot { width: 11px; height: 11px; border-radius: 50%; display: inline-block; }
</style>

# Detection skill across rainfall thresholds

The event-detection numbers on the [scorecard](./staged-skill) - the per-station median across the 172 BMKG stations - show POD falling from **0.78** (LS) to **0.65** after correction, which looks like a loss. It is not: it is a **crossover**. LS wins at drizzle by over-forecasting wet days; the corrected product gives that back in exchange for a calibrated wet-day frequency and higher skill from 20 mm/day upward, the ETCCDI very-heavy-rain threshold. Move across the threshold axis and watch it flip.

```js
import {STAGE_KEY, STAGE_COLORS} from "./components/util.js";
const curves = FileAttachment("data/threshold_curves.json").json();
```

```js
const METRIC_META = {
  pod: {label: "POD - probability of detection", better: "higher"},
  csi: {label: "CSI - critical success index", better: "higher"},
  far: {label: "FAR - false-alarm ratio", better: "lower"},
  ets: {label: "ETS - equitable threat score", better: "higher"},
  hss: {label: "HSS - Heidke skill score", better: "higher"},
  fbi: {label: "FBI - frequency bias (1 = perfect)", better: "near 1"}
};
const metric = view(Inputs.radio(
  new Map(Object.entries(METRIC_META).map(([k, v]) => [v.label.split(" - ")[0], k])),
  {value: "pod", label: "Metric"}));
```

```js
const showBand = view(Inputs.toggle({label: "Show station IQR band", value: false}));
```

```js
const STAGES = ["LS", "LSEQM", "LSEQM+DL"];
// `supported` is false at 100 and 150 mm: too few gauge exceedances to order the stages.
const long = STAGES.flatMap((s) => curves[STAGE_KEY[s]]
  .filter((d) => d.supported && d[metric] != null)
  .map((d) => ({stage: s, thr: d.thr, label: d.label, v: d[metric], lo: d[`${metric}_lo`], hi: d[`${metric}_hi`]})));
```

```js
display(Plot.plot({
  width,
  height: 430,
  marginRight: 20,
  x: {type: "log", label: "rainfall threshold (mm/day) →", domain: [0.85, 62],
      ticks: [1, 5, 10, 20, 50], tickFormat: (d) => `${d}`, grid: true},
  y: {label: `↑ ${METRIC_META[metric].label}`, grid: true,
      domain: metric === "fbi" ? [0, 1.55] : [0, 1]},
  color: {legend: true, domain: STAGES, range: STAGES.map((s) => STAGE_COLORS[s])},
  marks: [
    metric === "fbi" ? Plot.ruleY([1], {stroke: "#888", strokeDasharray: "4,3"}) : null,
    showBand ? Plot.areaY(long, {x: "thr", y1: "lo", y2: "hi", z: "stage", fill: "stage", fillOpacity: 0.10}) : null,
    Plot.line(long, {x: "thr", y: "v", z: "stage", stroke: "stage", strokeWidth: 2}),
    Plot.dot(long, {x: "thr", y: "v", fill: "stage", r: 3.5, stroke: "white", strokeWidth: 0.5,
      tip: true, channels: {stage: "stage", threshold: "label", value: "v"}}),
    metric === "far" ? Plot.text([{}], {frameAnchor: "top-right", dx: -6, dy: 6, text: ["lower is better ↓"], fontSize: 11, fill: "#999"}) : null
  ].filter(Boolean)
}));
```

```js
// dynamic per-metric reading of the curves, at drizzle (1 mm) and heavy (50 mm)
const val = (skey, thr) => { const r = curves[skey].find((d) => d.thr === thr); return r ? r[metric] : null; };
const pc = (x) => x == null ? "-" : `${(x * 100).toFixed(0)}%`;
const f2 = (x) => x == null ? "-" : x.toFixed(2);
const lsLo = val("ls", 1), dlLo = val("lseqmdl", 1), lsHi = val("ls", 50), dlHi = val("lseqmdl", 50);
const READ = {
  pod: html`At 1 mm/day LS leads by over-detecting drizzle (POD <b>${pc(lsLo)}</b> vs <b>${pc(dlLo)}</b>), but by 50 mm the corrected product overtakes it (<b>${pc(lsHi)}</b> vs <b>${pc(dlHi)}</b>) - it keeps the heavy-rain hits that matter.`,
  csi: html`CSI mirrors detection: LS is ahead at 1 mm (<b>${pc(lsLo)}</b> vs <b>${pc(dlLo)}</b>) yet falls behind at 50 mm (<b>${pc(lsHi)}</b> vs <b>${pc(dlHi)}</b>), where the corrected product's calibrated frequency pays off.`,
  far: html`False alarms drop after correction at every threshold (FAR <b>${pc(lsLo)}</b> → <b>${pc(dlLo)}</b> at 1 mm; <b>${pc(lsHi)}</b> → <b>${pc(dlHi)}</b> at 50 mm) - matching the wet-day frequency removes the spurious light-rain alarms LS raises. Lower is better.`,
  ets: html`ETS credits hits beyond chance. The stages run close at light rain; by 50 mm the lead goes to ${dlHi > lsHi ? "the corrected product" : "LS"} (<b>${f2(dlHi)}</b> vs <b>${f2(lsHi)}</b>).`,
  hss: html`HSS tracks ETS - near-identical at light thresholds, with any edge appearing only in the heavy tail (50 mm: <b>${f2(dlHi)}</b> vs <b>${f2(lsHi)}</b>).`,
  fbi: html`Frequency bias is the clearest tell: LS swings from over-forecasting (<b>${f2(lsLo)}</b> at 1 mm) to severe under-forecasting (<b>${f2(lsHi)}</b> at 50 mm), while LSEQM+DL holds near <b>1.0</b> across the range. Calibrated frequency is the whole point of the correction.`
};
display(html`<blockquote>${READ[metric]} LSEQM and LSEQM+DL track almost identically - the marginal step, not the CNN, drives the change.</blockquote>`);
```

<div class="note">
Skill is pooled as the cross-station median per dekad, then averaged over the 36 dekads (172 BMKG stations, daily totals paired on the archived day labels, that is the native h = 0 window). Above 50 mm/day the sample thins out fast: inside a given dekad-of-year window a station averages about <b>5</b> gauge exceedances at 50 mm over the whole 2001-2021 record, but only about <b>0.7</b> at 100 mm. WMO/TD-1485's ten-event minimum is then almost never met - only 6 of 5,899 station-dekad records qualify at 100 mm, with a median CSI of 0.000 for every stage, and none qualify at 150 mm. Both classes are therefore omitted from the curves rather than plotted as if they ranked the stages. Toggle the IQR band to see the across-station spread.
<br><br>
The pooled version of this trade-off is the event-detection pillar on the [staged scorecard](./staged-skill); the flat timing track it cannot fix is [the timing ceiling](./ceiling).
</div>
