---
title: The timing ceiling
---

<style>
.big { font-size: 1.7rem; font-weight: 700; display: block; line-height: 1.1; }
.callout { border-left: 3px solid var(--theme-foreground-focus); padding: 0.4rem 0 0.4rem 1rem; margin: 1rem 0; }
</style>

# Why correcting the distribution does not fix daily timing

Every stage lands daily Pearson *r* near **0.35** against CPC-UNI and near **0.24** against the independent BMKG stations. This is the expected behaviour of a marginal correction, not a shortcoming of this one. Each stage maps a pixel's value through a monotone non-decreasing function, so it introduces no rank reversals between days: if the satellite reported more rain on day *i* than day *j*, it still does after correction. Ordering is not preserved exactly, because dry-day matching ties about 43% of days at zero and ties can be created or broken there, but the effect on *r* is small. The day-by-day pairing is set by the satellite retrieval, and a marginal step cannot move rain from one day to its neighbour. You can reshape the **distribution** all you like - the **timing** does not move.

<div class="keyfinding">
<span class="kf-label">Key finding</span>
Against CPC-UNI, daily Pearson <i>r</i> moves just <b>0.005</b> across the three stages - from <b>0.343</b> (LS) to <b>0.345</b> (LSEQM) to <b>0.348</b> (LSEQM+DL), as a per-pixel spatial median averaged over the 36 dekads. Against the independent BMKG stations, under the identical recipe, it moves from <b>0.242</b> to <b>0.236</b> to <b>0.239</b>. The correction reshapes the distribution; it does not touch the timing, against either reference.
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
    raw ${rRaw.toFixed(3)} → corrected ${rCorr.toFixed(3)} · pinned close to the raw value
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

Daily *r* by stage, against both references. The recipe is the same in structure: a median taken across the reference's own units within each dekad - over the 19,393 land pixels for CPC-UNI, over the stations for BMKG - then averaged across the 36 dekads. CPC-UNI is the dataset the correction was fitted to, so it is the in-sample number; the 172 BMKG stations are held out of the fitting. Both are flat across the stages, which is the point of the page.

```js
const rcpc = headline.cpc.find((t) => /Pearson/.test(t.metric));
const rbmkg = headline.bmkg.find((t) => /Pearson/.test(t.metric));
const rbars = ["LS", "LSEQM", "LSEQM+DL"].flatMap((stage) => {
  const k = stage === "LSEQM+DL" ? "lseqmdl" : stage.toLowerCase();
  return [{stage, reference: "vs CPC-UNI (in-sample)", r: rcpc[k]},
          {stage, reference: "vs BMKG (independent)", r: rbmkg[k]}];
});
const s = headline.stats;
```

```js
display(Plot.plot({
  width: width,
  height: 300,
  marginLeft: 90,
  x: {axis: null},
  fx: {domain: ["LS", "LSEQM", "LSEQM+DL"], label: null},
  y: {domain: [0, 0.42], grid: true, label: "↑ daily Pearson r (dekad-averaged median)"},
  color: {legend: true, domain: ["vs CPC-UNI (in-sample)", "vs BMKG (independent)"], range: ["#4477aa", "#cc6677"]},
  marks: [
    Plot.barY(rbars, {fx: "stage", x: "reference", y: "r", fill: "reference", tip: true,
                      channels: {stage: "stage", reference: "reference", "Pearson r": "r"}}),
    Plot.text(rbars, {fx: "stage", x: "reference", y: "r", text: (d) => d.r.toFixed(3),
                      dy: -8, fontSize: 11}),
    Plot.ruleY([0])
  ]
}));
```

## The monthly Taylor

The limit is not an artefact of one summary statistic. Every product's Taylor position - all six, in all twelve months - sits at a similarly low correlation. The correction slides the cloud **along the standard-deviation axis** (LS under-spread → LSEQM/LSEQM+DL on the reference circle) but never **toward the correlation axis**. Each grey dot is one BMKG station (all six products pooled); the coloured dots are the per-product medians. The cloud concentrates in the low-correlation wedge - product medians between *r* = 0.20 and 0.24, most stations below 0.5 - so the limit is a property of the whole network, not of how the summary was taken. A thin tail of stations reaches higher, but no product median does. Pick a month:

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
A marginal (per-cell) correction is a monotone remap of each day's value; it cannot move rain from one day to its neighbour. So Pearson <i>r</i> stays <b>pinned close to the raw retrieval</b> - and on the Taylor diagram all six products cluster between <b>0.20</b> and <b>0.24</b>, taken as the per-station median of the dekadal correlation across the 171 stations that have one. What the correction fixes is the radius: on that same basis the standard-deviation ratio moves from <b>0.72</b> for LS to <b>1.00</b> for LSEQM+DL. The cloud moves out to the reference circle, never around toward the perfect corner.
</div>

<div class="note" style="margin-top:1.5rem">
The limit is real, and part of it is conventional rather than fundamental. Held to a fixed UTC-day pairing, no stage moves daily <i>r</i> away from <b>${s.r_flat_cpc}</b> against CPC-UNI or <b>${s.r_flat_bmkg}</b> against BMKG. How much of that is a day-label artefact can only be measured against BMKG, because CPC-UNI dates its totals to the UTC day exactly as IMERG does and so carries no offset: re-pairing IMERG-L to the BMKG gauge day raises the pooled daily correlation there from <b>${s.r_window_utc}</b> to <b>${s.r_window_local}</b> (see <a href="./window">the window diagnostic</a>). Beyond that, closing the remaining gap needs methods outside the marginal-correction family - the <a href="./paths">four paths</a>.
</div>
