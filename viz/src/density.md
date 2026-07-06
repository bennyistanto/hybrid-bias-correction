---
title: Station-density mask
---

<style>
.big { font-size: 1.7rem; font-weight: 700; display: block; line-height: 1.1; }
.maptile { width: 100%; max-width: 960px; display: block; border: 1px solid var(--theme-foreground-faintest); border-radius: 4px; }
.legendrow { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--theme-foreground-muted); max-width: 420px; margin-top: 6px; }
.legendbar { height: 12px; border-radius: 3px; border: 1px solid var(--theme-foreground-faintest); flex: 1; }
.eqn { font-family: var(--serif); font-size: 15px; background: var(--theme-background-alt); padding: 0.5rem 0.9rem; border-radius: 4px; display: inline-block; }
</style>

# The station-density confidence mask

The CNN refinement is not applied uniformly. A confidence field, built from the smoothed density of the CPC-UNI gauges, **gates the blend**: where gauges are dense the CNN contributes, where they are sparse the product falls back to pure LSEQM. This keeps the data-hungry step honest about where it has support.

```js
const density = FileAttachment("data/density.json").json();
const densityMap = FileAttachment("data/maps/density_confidence.png").url();
```

```js
const d = density;
const base = d.base_alpha;
```

## Where the gauges are - and where the CNN activates

Confidence over the 0.1&deg; land grid (Gaussian-smoothed, &sigma; ≈ ${d.sigma_km} km); the ${d.n_stations} gauges are overlaid as white dots. The network is heavily concentrated in Java - so is the confidence.

```js
display(html`<img class="maptile" src=${densityMap} alt="station-density confidence">`);
```

<div class="legendrow">
  <span>0 · pure LSEQM</span>
  <div class="legendbar" style="background:linear-gradient(to right,#440154,#414487,#2a788e,#22a884,#7ad151,#fde725)"></div>
  <span>${d.max_confidence} · max</span>
</div>

<div class="grid grid-cols-4" style="margin-top:1.5rem">
  <div class="card"><h2>Max confidence</h2><span class="big">${d.max_confidence}</span> of 1.0 (dense Java)</div>
  <div class="card"><h2>Max DL weight</h2><span class="big">${d.max_dl_weight_pct}%</span> where densest</div>
  <div class="card"><h2>Mean DL weight</h2><span class="big">${d.mean_dl_weight_pct}%</span> over all land</div>
  <div class="card"><h2>Land the CNN touches</h2><span class="big">${d.pct_dl_active}%</span> at any weight</div>
</div>

The mask never reaches full confidence: even the densest cell tops out at ${d.max_confidence}, so the CNN's share peaks near **${d.max_dl_weight_pct}%** in Java and averages just **${d.mean_dl_weight_pct}%** across Indonesia. The refinement is real but deliberately restrained.

## How confidence sets the blend

The blend weight is a straight line in confidence:

<div class="eqn">effective α = 1 − confidence × (1 − ${base}) &nbsp;&nbsp;→&nbsp;&nbsp; CNN weight = confidence × ${(1 - base).toFixed(2)}</div>

Confidence 0 gives α = 1.0 (pure LSEQM, no CNN); confidence 1 would give α = ${base} (30% CNN). The dashed line marks the mask's actual ceiling.

```js
const conf = d3.range(0, 1.0001, 0.02);
const blend = conf.flatMap((c) => [
  {c, series: "effective α (LSEQM weight)", v: 1 - c * (1 - base)},
  {c, series: "CNN weight", v: c * (1 - base)}
]);
```

```js
display(Plot.plot({
  width: width,
  height: 300,
  x: {label: "confidence →", domain: [0, 1], grid: true},
  y: {label: "↑ weight", domain: [0, 1.05], grid: true},
  color: {legend: true, domain: ["effective α (LSEQM weight)", "CNN weight"], range: ["#2f7d9e", "#e0913a"]},
  marks: [
    Plot.ruleX([d.max_confidence], {stroke: "#999", strokeDasharray: "4,3"}),
    Plot.text([{}], {frameAnchor: "top", dx: 8, text: [`mask ceiling ${d.max_confidence}`], fontSize: 10, fill: "#999", x: d.max_confidence, y: 1.02, textAnchor: "start"}),
    Plot.line(blend, {x: "c", y: "v", z: "series", stroke: "series", strokeWidth: 2}),
    Plot.crosshairX(blend, {x: "c", y: "v", color: "series"}),
    Plot.ruleY([0])
  ]
}));
```

## What if the saturation count changed?

Confidence saturates at `DENSITY_SATURATION_COUNT` smoothed stations per cell (default **${d.saturation_count}**). That threshold was set by judgement, not optimised - it is one of the [sensitivity parameters](./sensitivity). Lowering it lets sparser networks reach full confidence, activating more CNN; raising it is more conservative. Drag it to see the effect on the same mask.

```js
const sat = view(Inputs.range([1, 6], {value: d.saturation_count, step: 1, label: "Saturation count"}));
```

```js
const centers = d.hist_edges.slice(0, -1).map((e, i) => (e + d.hist_edges[i + 1]) / 2);
const totalN = d3.sum(d.hist_counts);
const maxCounts = d.hist_edges[d.hist_edges.length - 1];
const meanDL = d3.sum(centers.map((c, k) => d.hist_counts[k] * Math.min(c / sat, 1) * (1 - base))) / totalN * 100;
const maxDL = Math.min(maxCounts / sat, 1) * (1 - base) * 100;
// share of land whose CNN weight would clear 10%
const pctStrong = d3.sum(centers.map((c, k) => (Math.min(c / sat, 1) * (1 - base) >= 0.10 ? d.hist_counts[k] : 0))) / totalN * 100;
```

<div class="grid grid-cols-3">
  <div class="card"><h2>Mean CNN weight</h2><span class="big">${meanDL.toFixed(2)}%</span> over all land</div>
  <div class="card"><h2>Max CNN weight</h2><span class="big">${maxDL.toFixed(1)}%</span> in the densest cell</div>
  <div class="card"><h2>Land with CNN ≥ 10%</h2><span class="big">${pctStrong.toFixed(1)}%</span></div>
</div>

At the default of ${d.saturation_count}, the mean weight is ${d.mean_dl_weight_pct}% and almost no land clears a 10% CNN share - the conservative choice the thesis flags as a limitation. Dropping the count toward 1 concentrates more correction into the same Java-centred pattern rather than spreading it east, because the underlying gauge network, not the threshold, is what is sparse.
