---
title: Whole-domain & era
---

<style>
.dial-note { color: var(--theme-foreground-muted); font-size: 0.95rem; max-width: 46rem; }
</style>

# The window offset: whole-domain & era

The [calendar-window](./window) recovery rests on one window-offset diagnostic, re-run on every slice of the record. This page carries the depth behind it: when the offset emerged, that it holds in every timezone band and season, station by station, and across the whole 0.5&deg; domain.

## The satellite-era reveal

Why does the −23 h window unlock so much? Because the recovery **did not always exist**. Splitting the record by year shows the calendar-window gain appears only at the **2014/15 GPM-constellation transition**, once IMERG began resolving the diurnal cycle. Before that, the UTC-day *r* and the best-window *r* are almost identical and the optimal offset wanders; after it, they separate sharply and the offset locks to −23 h.

```js
const era = await FileAttachment("data/era_window.json").json();
const eraData = era.years.map((y, i) => ({year: y, r0: era.r0[i], pk: era.pk[i], hstar: era.hstar[i], hlo: era.hlo[i], hhi: era.hhi[i]}));
```

<div class="dial-note" style="font-size:13px">
<span style="color:#5a189a">■</span> peak <i>r</i> (window-aligned, h = h★) &nbsp; <span style="color:#d62728">■</span> <i>r</i> at h = 0 (UTC-day archive) &nbsp; <span style="color:#d9b3ff">■</span> recoverable by window alignment
</div>

```js
display(Plot.plot({
  width,
  height: 300,
  marginLeft: 46,
  x: {domain: [2000.5, 2021.5], label: null, tickFormat: "d", ticks: d3.range(2001, 2022, 2)},
  y: {domain: [0, 0.7], grid: true, label: "↑ Pearson r (all months pooled)"},
  marks: [
    Plot.areaY(eraData, {x: "year", y1: "r0", y2: "pk", fill: "#d9b3ff", fillOpacity: 0.5}),
    Plot.ruleX([era.gpm_split], {stroke: "#888888", strokeDasharray: "4,3"}),
    Plot.line(eraData, {x: "year", y: "pk", stroke: "#5a189a", strokeWidth: 2, marker: "circle", tip: true, channels: {year: "year", "peak r": "pk"}}),
    Plot.line(eraData, {x: "year", y: "r0", stroke: "#d62728", strokeWidth: 2, marker: "circle", tip: true, channels: {year: "year", "r at h=0": "r0"}}),
    Plot.text([{x: 2014.75}], {x: "x", y: 0.03, text: ["GPM era →"], fontSize: 10, fill: "#555555", textAnchor: "start"}),
    Plot.text([{x: 2014.25}], {x: "x", y: 0.03, text: ["← TRMM-era input"], fontSize: 10, fill: "#555555", textAnchor: "end"})
  ]
}));
```

```js
display(Plot.plot({
  width,
  height: 210,
  marginLeft: 46,
  x: {domain: [2000.5, 2021.5], label: "year", tickFormat: "d", ticks: d3.range(2001, 2022, 2)},
  y: {domain: [-40, 5], grid: true, label: "↑ optimal offset h★ (hours)", ticks: [-30, -20, -10, 0]},
  marks: [
    Plot.areaY(eraData, {x: "year", y1: "hlo", y2: "hhi", fill: "#bbbbbb", fillOpacity: 0.3}),
    Plot.ruleY([era.convention_offset], {stroke: "#5a189a", strokeDasharray: "2,2"}),
    Plot.ruleX([era.gpm_split], {stroke: "#888888", strokeDasharray: "4,3"}),
    Plot.line(eraData, {x: "year", y: "hstar", stroke: "#333333", strokeWidth: 1.6, marker: "circle", tip: true, channels: {year: "year", "h★": "hstar"}}),
    Plot.text([{x: 2001}], {x: "x", y: -27, text: ["convention ≈ −23 h"], fill: "#5a189a", fontSize: 10, textAnchor: "start"})
  ]
}));
```

<div class="keyfinding">
<span class="kf-label">Finding</span>
Before the 2014/15 GPM transition the UTC-day <i>r</i> and the best-window <i>r</i> sit almost on top of each other, and the optimal offset <b>h★</b> wanders - the sub-daily timing signal is undetectable. After it, the window-aligned <i>r</i> pulls clear of the UTC-day <i>r</i> (the shaded gain, <b>0.20</b> → <b>0.57</b>: daily <i>r</i> against the independent BMKG stations, a single correlation pooled over all station-days of the GPM era, 2015-2021) and h★ locks to <b>−22/−23 h</b>. The calendar-window recovery only exists once IMERG resolves the diurnal cycle.
</div>

## Season and timezone

Zoom into the GPM era (2015-2021) and the −23 h peak holds in **every timezone band** and **every season**. Panel (a) is the mean *r(h)* per band with the **inter-quartile range across stations** shaded; panel (b) tracks the peak offset across the twelve three-month running seasons.

```js
const wg = await FileAttachment("data/window_gpm.json").json();
const TZ_COLORS = {"WIB (UTC+7)": "#1f77b4", "WITA (UTC+8)": "#ff7f0e", "WIT (UTC+9)": "#2ca02c"};
const TZ_NAME = {7: "WIB (UTC+7)", 8: "WITA (UTC+8)", 9: "WIT (UTC+9)"};
```

```js
const rhLong = wg.bands.flatMap((b) => wg.h.map((hh, i) => ({band: b.name, h: hh, mean: b.mean[i], q25: b.q25[i], q75: b.q75[i]})));
const rhPeaks = wg.bands.map((b) => ({band: b.name, h: b.peakH, r: b.peakR}));
display(Plot.plot({
  width,
  height: 340,
  x: {domain: [-48, 48], label: "hour offset h →", ticks: d3.range(-48, 49, 12), grid: true},
  y: {domain: [-0.05, 0.75], label: "↑ Pearson r (mean per band, IQR shaded)", grid: true},
  color: {legend: true, domain: Object.keys(TZ_COLORS), range: Object.values(TZ_COLORS)},
  marks: [
    Plot.areaY(rhLong, {x: "h", y1: "q25", y2: "q75", fill: "band", fillOpacity: 0.13, z: "band"}),
    Plot.ruleX([0, -23], {stroke: "#888888", strokeDasharray: "3,3"}),
    Plot.line(rhLong, {x: "h", y: "mean", stroke: "band", z: "band", strokeWidth: 2}),
    Plot.dot(rhPeaks, {x: "h", y: "r", fill: "band", r: 5, stroke: "black", strokeWidth: 0.6}),
    Plot.crosshairX(rhLong, {x: "h", y: "mean", color: "band"}),
    Plot.text([{h: -23}], {x: "h", y: 0.72, text: ["h★ = −23 h"], fill: "#5a189a", fontSize: 10, textAnchor: "middle"})
  ]
}));
```

```js
const seasonLong = [7, 8, 9].flatMap((z) => wg.season.order.map((s, i) => ({series: TZ_NAME[z], season: s, h: wg.season.bands[String(z)][i]})))
  .concat(wg.season.order.map((s, i) => ({series: "pooled", season: s, h: wg.season.pooled[i]})));
display(Plot.plot({
  width,
  height: 260,
  marginBottom: 58,
  x: {domain: wg.season.order, label: "three-month running season", tickRotate: -45},
  y: {domain: [-30, -14], label: "↑ optimal offset h★ (hours)", grid: true},
  color: {legend: true, domain: ["WIB (UTC+7)", "WITA (UTC+8)", "WIT (UTC+9)", "pooled"], range: ["#1f77b4", "#ff7f0e", "#2ca02c", "#222222"]},
  marks: [
    Plot.ruleY([-23], {stroke: "#5a189a", strokeDasharray: "2,2"}),
    Plot.line(seasonLong, {x: "season", y: "h", stroke: "series", z: "series", strokeWidth: 1.8, marker: "circle", tip: true, channels: {series: "series", season: "season", "h★": "h"}})
  ]
}));
```

## Per-station offset & lift

Every station's own best offset, and the correlation it unlocks. Median h★ is **${wg.summary.median_hstar} h** (IQR ${wg.summary.iqr_hstar[0]} to ${wg.summary.iqr_hstar[1]}). The lift from the UTC day to **each station's own** best offset, taken per station and then median over the 178 stations, is **Δr = ${wg.summary.median_lift}** (IQR 0.318 to 0.444), putting the typical station's peak daily *r* against BMKG at **${wg.summary.median_peakr}** (IQR 0.51 to 0.62).

This is a *different* quantity from the **0.20 → 0.57** on the [calendar-window page](./window), which is one correlation pooled over all station-days at a single fixed offset of −23 h. The two land close together here, but they are not the same measurement and should not be quoted for each other.

```js
const idnAdm1w = FileAttachment("data/idn_adm1.geojson").json();
const neighboursw = FileAttachment("data/neighbours.geojson").json();
```

```js
display(Plot.plot({
  width,
  marginLeft: 6, marginRight: 6, marginTop: 6, marginBottom: 6,
  projection: {type: "equirectangular", domain: idnAdm1w, inset: 6},
  color: {legend: true, type: "linear", scheme: "rdylbu", reverse: true, domain: [-28, -12], clamp: true, label: "h★ (h from UTC midnight)"},
  marks: [
    Plot.geo(neighboursw, {fill: "#ececec", stroke: "#d4d4d4", strokeWidth: 0.4}),
    Plot.geo(idnAdm1w, {fill: "#f7f7f2", stroke: "#cccccc", strokeWidth: 0.4}),
    Plot.dot(wg.stations, {x: "lon", y: "lat", fill: "hstar", r: 3.5, stroke: "black", strokeWidth: 0.3,
      tip: true, channels: {Station: "name", "h★": "hstar", "r at h=0": "r0", "peak r": "peakr", lift: "lift"}})
  ]
}));
```

```js
const ordered = [...wg.stations].sort((a, b) => a.lon - b.lon).map((s, i) => ({...s, rank: i, tzname: TZ_NAME[s.tz]}));
display(Plot.plot({
  width,
  height: 520,
  marginLeft: 20,
  x: {domain: [-0.05, 0.9], label: "Pearson r (GPM era) - grey = UTC day, colour = matched window →", grid: true},
  y: {axis: null, domain: [-1, ordered.length]},
  color: {legend: true, domain: ["WIB (UTC+7)", "WITA (UTC+8)", "WIT (UTC+9)"], range: ["#1f77b4", "#ff7f0e", "#2ca02c"]},
  marks: [
    Plot.ruleY(ordered, {y: "rank", x1: "r0", x2: "peakr", stroke: "#cccccc", strokeWidth: 0.6}),
    Plot.dot(ordered, {x: "r0", y: "rank", fill: "#888888", r: 1.8}),
    Plot.dot(ordered, {x: "peakr", y: "rank", fill: "tzname", r: 2.6, tip: true, channels: {Station: "name", r0: "r0", "peak r": "peakr", lift: "lift"}})
  ]
}));
```

<div class="dial-note" style="font-size:13px">Stations ordered west → east (bottom → top) by longitude. Each grey dot is the UTC-day *r*; the coloured dot is the window-matched *r*; the bar between them is the per-station lift.</div>

## Gridded whole-domain confirmation

The station result holds over the **whole 0.5° domain**, not just at the gauges. Re-window the half-hourly IMERG-L against the gridded CPC-UNI analysis: at CPC's native UTC labels the best offset is **≈ 0 h everywhere** (yellow); harmonise CPC to the local-observation day and it jumps to **≈ −23 h everywhere** (purple). Uniform and era-stable.

```js
const gnative = FileAttachment("data/maps/gridded_hstar_native.png").url();
const gharm = FileAttachment("data/maps/gridded_hstar_harmonised.png").url();
const wgrid = await FileAttachment("data/window_gridded.json").json();
const imgStyle = "width:100%;display:block;border:1px solid var(--theme-foreground-faintest);border-radius:4px;margin:0.3rem 0";
```

```js
display(html`<div style="max-width:440px;margin:0.4rem 0 0.8rem">
  <div style="height:13px;border-radius:3px;border:1px solid var(--theme-foreground-faintest);background:linear-gradient(to right,#440154,#414487,#2a788e,#22a884,#7ad151,#fde725)"></div>
  <div style="position:relative;height:15px;font-size:11px;color:var(--theme-foreground-muted)">
    <span style="position:absolute;left:0">−26</span>
    <span style="position:absolute;left:10.7%;transform:translateX(-50%)">−23</span>
    <span style="position:absolute;left:50%;transform:translateX(-50%)">−12</span>
    <span style="position:absolute;left:92.9%;transform:translateX(-50%)">0</span>
    <span style="position:absolute;right:0">+2</span>
  </div>
  <div style="font-size:11px;color:var(--theme-foreground-muted)">h★ (hours from UTC midnight), viridis: purple ≈ −23 (local day), yellow ≈ 0 (UTC day). CPC-UNI 0.5&deg; grid.</div>
</div>`);
```

<h3>(a) CPC at native UTC labels - h★ ≈ 0</h3>

```js
display(html`<img src=${gnative} style=${imgStyle} alt="native h*">`);
```

<h3>(b) CPC harmonised to the local-observation day - h★ ≈ −23 h</h3>

```js
display(html`<img src=${gharm} style=${imgStyle} alt="harmonised h*">`);
```

## Gridded r(h): gauged vs whole domain

Pooling the offset sweep confirms the two levels: over the **134 gauge-hosting cells** the sweep peaks at **0.566**, against **0.335** over the **whole domain**, diluted by the data-sparse east. Both are gridded CPC-UNI cell values, not the BMKG per-station figure, and both peak at **h = +1** rather than at the −23 h the gauges show. Each sits within 0.003 of its value at the archived **h = 0** (0.564 and 0.334), which is the point: CPC-UNI dates its totals to the UTC day as IMERG does, so re-windowing barely moves it. The peak offset agrees to within an hour between the GPM and TRMM-input eras.

```js
display(html`<div style="display:flex;flex-wrap:wrap;gap:12px;font-size:12px;margin-bottom:4px">${wgrid.curves.map((c) => html`<span style="display:inline-flex;align-items:center;gap:5px"><svg width=24 height=8><line x1=0 y1=4 x2=24 y2=4 stroke=${c.color} stroke-width=${Math.min(c.width, 2.5)} stroke-dasharray=${c.dash ? "4,3" : "0"}></line></svg>${c.label}</span>`)}</div>`);
```

```js
display(Plot.plot({
  width,
  height: 360,
  x: {domain: [-48, 48], label: "window offset h (hours) →", ticks: d3.range(-48, 49, 12), grid: true},
  y: {domain: [0.05, 0.72], label: "↑ pooled Pearson r vs CPC-UNI", grid: true},
  marks: [
    Plot.ruleX([0, -23], {stroke: "#999999", strokeDasharray: "3,3"}),
    ...wgrid.curves.map((c) => Plot.line(
      c.r.map((r, i) => ({h: wgrid.h[i], r})).filter((d) => d.r != null),
      {x: "h", y: "r", stroke: c.color, strokeWidth: c.width, strokeDasharray: c.dash ? "5,4" : null})),
    Plot.crosshairX(wgrid.curves.flatMap((c) => c.r.map((r, i) => ({label: c.label, h: wgrid.h[i], r})).filter((d) => d.r != null)), {x: "h", y: "r"}),
    Plot.text([{h: -23}], {x: "h", y: 0.09, text: ["h = −23"], fontSize: 9, fill: "#555", textAnchor: "end", dx: -2}),
    Plot.text([{h: 0}], {x: "h", y: 0.09, text: ["h = 0"], fontSize: 9, fill: "#555", textAnchor: "start", dx: 2})
  ]
}));
```

<div class="keyfinding">
<span class="kf-label">Finding</span>
Native CPC-UNI shares IMERG-L's UTC day, so its peak sits at <b>h = 0</b>; harmonised to the local day it moves to <b>h = −23</b> - the same clock offset the gauges show. Gauged cells reach <b>${wgrid.summary.gauged_peak}</b>, the whole domain <b>${wgrid.summary.whole_peak}</b>. Solid = GPM era, dashed = TRMM-input era: the offset agrees to within an hour across the two, so the calendar mismatch is a fixed convention, not a retrieval artefact.
</div>

<div class="note" style="margin-top: 1.5rem">
Every curve here is computed with the same window-offset diagnostic that produced the [calendar-window dial](./window) - the recovery is real, it holds in every timezone band, season and satellite era tested as well as across the whole domain, and it is reproducible from the released code.
</div>
