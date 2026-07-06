---
title: Start here
---

<style>
.hero { margin: 1rem 0 1.5rem; }
.hero h1 { font-size: 2.1rem; line-height: 1.1; max-width: 34rem; }
.hero h2 { font-size: 1rem; font-weight: 400; color: var(--theme-foreground-muted); max-width: 40rem; }
.big { font-size: 1.9rem; font-weight: 700; display: block; line-height: 1.1; }
.toc { list-style: none; padding: 0; margin: 0.3rem 0 0; }
.toc li { padding: 3px 0; font-size: 14px; }
.toc a { text-decoration: none; }
.toc .num { color: var(--theme-foreground-muted); font-variant-numeric: tabular-nums; margin-right: 6px; }
</style>

<div class="hero">
  <h1>Hybrid bias correction of daily satellite precipitation over Indonesia</h1>
  <h2>An interactive look at what a four-stage bias correction does to daily satellite rainfall - what it fixes, what it structurally cannot, and where the ceiling turns out to be a fixable calendar-window artefact.</h2>
</div>

This dashboard presents the findings of a **hybrid bias-correction framework** for daily satellite precipitation, worked through the Indonesian archipelago as a case study. Satellite rainfall from IMERG-L is corrected toward gauge observations in four stages - Linear Scaling, Empirical Quantile Mapping, a Generalized Pareto tail, and a Convolutional Neural Network refinement (together, **LSEQM+DL**) - then validated against an independent network of BMKG stations.

The pages let you explore what the correction achieves and where its limits lie: the daily **distribution** moves onto the gauge, the day-by-day **timing** does not, and much of that timing ceiling turns out to be a fixable **calendar-window** artefact rather than a fundamental limit. Every figure is computed from the same processing pipeline.

```js
const headline = FileAttachment("data/headline.json").json();
const stations = FileAttachment("data/stations.json").json();
const idnAdm1 = FileAttachment("data/idn_adm1.geojson").json();
const neighbours = FileAttachment("data/neighbours.geojson").json();
```

```js
const s = headline.stats;
```

<div class="grid grid-cols-4">
  <div class="card"><h2>BMKG stations validated</h2><span class="big">${s.stations}</span> of ${s.archived} archived</div>
  <div class="card"><h2>IMERG-L land pixels</h2><span class="big">${s.pixels.toLocaleString("en")}</span> at 0.1&deg; (~11 km)</div>
  <div class="card"><h2>Record</h2><span class="big">${s.dekads}</span> dekads &middot; ${s.period_cpc}</div>
  <div class="card"><h2>Reproducible on Colab</h2><span class="big">${s.repro_min}</span> min for the Bali subdomain</div>
</div>

## What the correction fixes - and what it structurally cannot

The corrected product moves to the gauge **distribution**, but the day-by-day **timing** does not improve: Pearson *r* stays near **${s.r_flat}** across every stage. Most of that ceiling, though, is a fixable **calendar-window** artefact - re-aggregating IMERG-L to the local-day window lifts *r* against BMKG from **${s.r_window_utc}** to **${s.r_window_local}** (at ${s.offset_h} h).

## Study area & data coverage

180 BMKG stations (7 island groups) over the 0.1&deg; IMERG footprint; hover a station for its details.

```js
display(Plot.plot({
  width,
  marginLeft: 6,
  marginRight: 6,
  marginTop: 6,
  marginBottom: 6,
  projection: {type: "equirectangular", domain: idnAdm1, inset: 6},
  color: {legend: true, label: "Region"},
  marks: [
    Plot.geo(neighbours, {fill: "#ececec", stroke: "#d4d4d4", strokeWidth: 0.4}),
    Plot.geo(idnAdm1, {fill: "#eef6ff", stroke: "#a9c4da", strokeWidth: 0.4}),
    Plot.dot(stations, {
      x: "lon", y: "lat", fill: "region", r: 3.2, stroke: "white", strokeWidth: 0.4,
      tip: true, channels: {Station: "name", Province: "prov", Zone: "tz", "Elev (m)": "elev"}
    })
  ]
}));
```

## How to read this dashboard

A slide deck can carry the introduction and methods; these pages carry the findings. Move through them with the sidebar.

<div class="grid grid-cols-3">
  <div class="card">
    <b>The correction</b> <span style="color:var(--theme-foreground-muted)">- what it does</span>
    <ul class="toc">
      <li><a href="./staged-skill">What it fixes</a> - the staged scorecard</li>
      <li><a href="./skill">Detection by threshold</a> - the POD crossover</li>
      <li><a href="./spatial">Spatial quality</a> - stage-morph QA maps</li>
      <li><a href="./stations">Stations &amp; seasons</a> - Taylor + drill-down</li>
    </ul>
  </div>
  <div class="card">
    <b>The ceiling - and the fix</b>
    <ul class="toc">
      <li><a href="./ceiling">The timing ceiling</a> - why <i>r</i> ≈ 0.34</li>
      <li><a href="./window">The calendar window</a> - the convention dial</li>
      <li><a href="./window-detail">Whole-domain &amp; era</a> - the evidence behind it</li>
    </ul>
  </div>
  <div class="card">
    <b>Trusting &amp; using it</b>
    <ul class="toc">
      <li><a href="./sensitivity">Sensitivity</a> - the three parameters</li>
      <li><a href="./density">Station-density mask</a> - where the CNN activates</li>
      <li><a href="./applications">Who it serves</a> - well- vs poorly-served</li>
      <li><a href="./reproducibility">Reproducibility</a> - Colab runtime</li>
      <li><a href="./paths">Raising the ceiling</a> - four paths forward</li>
    </ul>
  </div>
</div>

<div class="note" style="margin-top:2rem; font-size:13px; color:var(--theme-foreground-muted)">
Built with <a href="https://observablehq.com/framework/" target="_blank" rel="noopener">Observable Framework</a> - a static, self-contained data app. Data: IMERG-L (GPM / NASA), CPC-UNI (NOAA), and the BMKG station network. Open code &amp; data: <a href="https://github.com/bennyistanto/hybrid-bias-correction" target="_blank" rel="noopener">GitHub</a> &middot; <a href="https://doi.org/10.5281/zenodo.20287847" target="_blank" rel="noopener">Zenodo</a>.
</div>
