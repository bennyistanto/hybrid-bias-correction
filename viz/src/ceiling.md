---
title: The timing ceiling
---

<style>
.big { font-size: 1.7rem; font-weight: 700; display: block; line-height: 1.1; }
.callout { border-left: 3px solid var(--theme-foreground-focus); padding: 0.4rem 0 0.4rem 1rem; margin: 1rem 0; }
</style>

# The r ≈ 0.34 timing ceiling

Every stage lands Pearson *r* near **0.34** against the gauge. This is not a shortcoming of the correction - it is the **predicted** behaviour of any marginal correction. The pipeline's quantile mapping is strictly monotonic, so if the satellite reports more rain on day *i* than day *j* before correction, it still does after. The day-by-day pairing is fixed by the satellite; no marginal step can move rain from one day to its neighbour. You can reshape the **distribution** all you like - the **timing** does not move.

<div class="keyfinding">
<span class="kf-label">Key finding</span>
Across all three correction stages, pooled daily Pearson <i>r</i> moves just <b>0.005</b> - from <b>0.343</b> (LS) to <b>0.345</b> (LSEQM) to <b>0.348</b> (LSEQM+DL). The correction reshapes the distribution; it does not touch the timing.
</div>

```js
import {STAGE_COLORS} from "./components/util.js";
const headline = FileAttachment("data/headline.json").json();
```

## See it on a synthetic pixel

A synthetic satellite-and-gauge day series (illustrative, as in the thesis bound schematic). Drag the quantile-mapping strength: the corrected series slides onto the gauge distribution - the distribution mismatch (KS) collapses - but its correlation with the gauge barely twitches.

```js
function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const rand = mulberry32(7);
function randn() { let u = 0, v = 0; while (!u) u = rand(); while (!v) v = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
const N = 160;
const gauge = [], sat = [];
for (let i = 0; i < N; i++) {
  const latent = randn();
  gauge.push(Math.max(0, 4 + 3.2 * latent + 1.2 * randn()));   // gauge mm/day
  sat.push(Math.max(0, 1.5 + 1.4 * latent + 2.6 * randn()));   // satellite: correlated, biased low, noisier
}
// quantile-map target: satellite ranks mapped onto the gauge distribution
const order = sat.map((_, i) => i).sort((p, q) => sat[p] - sat[q]);
const gsort = [...gauge].sort(d3.ascending);
const qmap = new Array(N);
order.forEach((i, rank) => { qmap[i] = gsort[rank]; });

function pearson(a, b) {
  const ma = d3.mean(a), mb = d3.mean(b);
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < a.length; i++) { const x = a[i] - ma, y = b[i] - mb; num += x * y; da += x * x; db += y * y; }
  return num / Math.sqrt(da * db);
}
function ksStat(a, b) {
  const as = [...a].sort(d3.ascending), bs = [...b].sort(d3.ascending);
  let d = 0;
  for (const v of [...as, ...bs]) d = Math.max(d, Math.abs(d3.bisectRight(as, v) / as.length - d3.bisectRight(bs, v) / bs.length));
  return d;
}
```

```js
const lambda = view(Inputs.range([0, 1], {value: 0, step: 0.05, label: "Quantile-mapping strength λ"}));
```

```js
const corrected = sat.map((v, i) => (1 - lambda) * v + lambda * qmap[i]);
const rRaw = pearson(sat, gauge);
const rCorr = pearson(corrected, gauge);
const ks = ksStat(corrected, gauge);
```

<div class="grid grid-cols-2">
  <div class="card">
    <h2>Correlation with gauge</h2>
    <span class="big">${rCorr.toFixed(3)}</span>
    raw ${rRaw.toFixed(3)} → corrected ${rCorr.toFixed(3)} · essentially unchanged
  </div>
  <div class="card">
    <h2>Distribution mismatch (KS)</h2>
    <span class="big">${ks.toFixed(3)}</span>
    ${lambda === 0 ? "uncorrected" : `↓ from ${ksStat(sat, gauge).toFixed(3)} at λ = 0`}
  </div>
</div>

```js
display(Plot.plot({
  width: width,
  height: 340,
  x: {label: "corrected satellite (mm/day) →", grid: true},
  y: {label: "↑ gauge (mm/day)", grid: true},
  marks: [
    Plot.dot(corrected.map((v, i) => ({x: v, y: gauge[i]})), {x: "x", y: "y", r: 3, fill: "#2f7d9e", fillOpacity: 0.7,
      tip: true, channels: {"corrected sat (mm)": "x", "gauge (mm)": "y"}}),
    Plot.linearRegressionY(corrected.map((v, i) => ({x: v, y: gauge[i]})), {x: "x", y: "y", stroke: "#b2182b"})
  ]
}));
```

<div class="callout">
Slide λ from 0 to 1: KS falls toward 0 (the corrected distribution meets the gauge) while <b>r holds near ${rRaw.toFixed(2)}</b> the whole way. Correcting the margin cannot buy timing that the satellite never measured.
</div>

## The same thing on the real numbers

Pooled daily *r* against the 172 BMKG stations, by stage - flat by construction. The dashed line is what the **same data** reaches once re-aggregated to the local-day window: the real headroom is not in the correction stages, it is in the [calendar-window convention](./window).

```js
const rrow = headline.temporal.find((t) => /Pearson/.test(t.metric));
const rbars = [["LS", rrow.ls], ["LSEQM", rrow.lseqm], ["LSEQM+DL", rrow.lseqmdl]].map(([stage, r]) => ({stage, r}));
const s = headline.stats;
```

```js
display(Plot.plot({
  width: width,
  height: 300,
  marginLeft: 90,
  x: {domain: ["LS", "LSEQM", "LSEQM+DL"], label: null},
  y: {domain: [0, 0.65], grid: true, label: "↑ daily Pearson r vs BMKG"},
  color: {domain: ["LS", "LSEQM", "LSEQM+DL"], range: ["LS", "LSEQM", "LSEQM+DL"].map((k) => STAGE_COLORS[k])},
  marks: [
    Plot.barY(rbars, {x: "stage", y: "r", fill: "stage", tip: true, channels: {stage: "stage", "Pearson r": "r"}}),
    Plot.text(rbars, {x: "stage", y: "r", text: (d) => d.r.toFixed(3), dy: -8, fontSize: 12}),
    Plot.ruleY([s.r_window_local], {stroke: "#b2182b", strokeDasharray: "5,4"}),
    Plot.text([{}], {frameAnchor: "top-right", dx: -6, dy: -2, y: s.r_window_local, text: [`${s.r_window_local} with local-day window →`], fontSize: 11, fill: "#b2182b", textAnchor: "end"}),
    Plot.ruleY([0])
  ]
}));
```

## The monthly Taylor

The bound is not a one-off pooled number. Every product's Taylor position - all six, in all twelve months - sits at the same low correlation. The correction slides the cloud **along the standard-deviation axis** (LS under-spread → LSEQM/LSEQM+DL on the reference circle) but never **toward the correlation axis**. Each grey dot is one BMKG station (all six products pooled); the coloured dots are the per-product medians. The cloud concentrates in the low-correlation wedge - a median near *r* ≈ 0.23, most stations below 0.5 - so the ceiling is a property of the whole network, not an artefact of pooling. A thin tail of stations reaches higher, but no product median does. Pick a month:

```js
const mtaylor = await FileAttachment("data/monthly_taylor.json").json();
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const monthSel = view(Inputs.select(["All year", ...MONTHS], {value: "All year", label: "Month"}));
const showSpread = view(Inputs.toggle({label: "Show per-station spread", value: true}));
```

```js
const monthKey = monthSel === "All year" ? "all" : String(MONTHS.indexOf(monthSel) + 1);
const md = mtaylor.months[monthKey];
const mpts = mtaylor.products.filter((p) => md[p.key].r != null).map((p) => {
  const r = md[p.key].r, sdr = md[p.key].sdr;
  return {key: p.key, label: p.label, color: p.color, r, sdr, tx: sdr * r, ty: sdr * Math.sqrt(Math.max(0, 1 - r * r))};
});
const cloudPts = showSpread
  ? (md.cloud ?? []).map(([r, sdr]) => ({tx: sdr * r, ty: sdr * Math.sqrt(Math.max(0, 1 - r * r))}))
  : [];
const maxR = 2.0, CORR = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99], SD = [0.5, 1.0, 1.5, 2.0];
const arcT = d3.range(0, Math.PI / 2 + 0.001, 0.02);
const stdArcs = SD.flatMap((s) => arcT.map((t) => ({s, x: s * Math.cos(t), y: s * Math.sin(t)})));
const refArc = arcT.map((t) => ({x: Math.cos(t), y: Math.sin(t)}));
const corrRays = CORR.map((r) => ({r, x: maxR * r, y: maxR * Math.sqrt(1 - r * r)}));
const rayLines = corrRays.flatMap((d) => [{r: d.r, x: 0, y: 0}, {r: d.r, x: d.x, y: d.y}]);
const sdLabels = SD.map((s) => ({s, x: s * Math.cos(Math.PI / 4), y: s * Math.sin(Math.PI / 4)}));
```

```js
display(Plot.plot({
  width: Math.min(width, 560),
  height: Math.min(width, 560),
  aspectRatio: 1,
  x: {domain: [0, maxR], label: "standard deviation (normalized) →", ticks: SD, grid: true},
  y: {domain: [0, maxR], label: "standard deviation (normalized) ↑", ticks: SD, grid: true},
  color: {legend: true, domain: mtaylor.products.map((p) => p.label), range: mtaylor.products.map((p) => p.color)},
  marks: [
    Plot.line(stdArcs, {x: "x", y: "y", z: "s", stroke: "#dddddd", strokeWidth: 0.8}),
    Plot.line(refArc, {x: "x", y: "y", stroke: "#9ecae1", strokeWidth: 1.2, strokeDasharray: "4,3"}),
    Plot.line(rayLines, {x: "x", y: "y", z: "r", stroke: "#eeeeee", strokeWidth: 0.8}),
    Plot.text(corrRays, {x: "x", y: "y", text: (d) => `${d.r}`, dx: 7, dy: -5, fontSize: 10, fill: "#777777"}),
    Plot.text([{x: maxR * Math.cos(0.87), y: maxR * Math.sin(0.87)}], {x: "x", y: "y", text: ["correlation"], fontSize: 11, fill: "#777777", rotate: -38, dx: 4, dy: -8}),
    Plot.text(sdLabels, {x: "x", y: "y", text: (d) => d.s.toFixed(1), fontSize: 9, fill: "#9a9a9a", dx: -7, dy: 9}),
    Plot.dot(cloudPts, {x: "tx", y: "ty", r: 1.8, fill: "#c9c9c9", fillOpacity: 0.5}),
    Plot.dot([{x: 1, y: 0}], {symbol: "star", r: 9, fill: "#111111"}),
    Plot.dot(mpts, {x: "tx", y: "ty", fill: "label", r: 7, stroke: "white", strokeWidth: 1,
      tip: true, channels: {Product: "label", r: "r", "std ratio": "sdr"}}),
    Plot.text(mpts, {x: "tx", y: "ty", text: (d) => d.label, fill: "label", dy: -12, fontSize: 9, fontWeight: 600})
  ]
}));
```

<div class="keyfinding">
<span class="kf-label">Headline proposition</span>
A marginal (per-cell) correction is a monotonic remap of each day's value; it cannot move rain from one day to its neighbour. So Pearson <i>r</i> is <b>bounded by the raw retrieval</b> - and on the Taylor diagram every product, every month, clusters near <b>r ≈ 0.22</b>. What the correction fixes is the radius: the standard-deviation ratio goes from LS <b>0.72</b> to LSEQM+DL <b>≈ 1.0</b>. The cloud moves out to the reference circle, never around toward the perfect corner.
</div>

<div class="note" style="margin-top:1.5rem">
The ceiling is real but conventional, not fundamental. For a fixed UTC-day pairing the stages cannot beat ~<b>${s.r_flat}</b>; re-labelling to the local day lifts the <i>same</i> product to <b>${s.r_window_local}</b> (see <a href="./window">5.2</a>). Beyond that, closing the gap needs methods outside the marginal-correction family - the <a href="./paths">four paths</a>.
</div>
