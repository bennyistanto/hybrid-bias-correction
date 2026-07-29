---
title: The calendar window
---

<style>
.big { font-size: 2.1rem; font-weight: 700; display: block; line-height: 1.1; }
.dial-note { color: var(--theme-foreground-muted); font-size: 0.95rem; max-width: 46rem; }
</style>

# The calendar-window discovery

Daily satellite-gauge correlation is **convention-dependent**. IMERG-L is dated by the **UTC** calendar day; the BMKG gauges report on the **local-day** morning-observation window. So the two references peak at *different* aggregation-window offsets. Drag the dial and watch the peaks separate - the low daily correlation is a **clock** problem, not a rainfall problem.

<div class="keyfinding">
<span class="kf-label">Key finding</span>
Re-aggregating IMERG-L from the UTC day to the local-day window (offset <b>−23 h</b>) lifts the daily correlation against BMKG from <b>0.20</b> to <b>0.57</b> - a single correlation pooled over all station-days of the GPM era, 2015-2021 - with no change to the correction itself. A large part of the daily timing gap against the station network is therefore a calendar artefact rather than a limit of the retrieval. Note this applies to the BMKG comparison only: CPC-UNI dates its daily totals to the UTC day exactly as IMERG does, so it carries no offset and the same re-windowing degrades it instead.
</div>

```js
import {fmt, WINDOW_COLORS} from "./components/util.js";
const windowCurves = FileAttachment("data/window_curves.json").json();
const wg = await FileAttachment("data/window_gpm.json").json();
```

```js
const h = view(Inputs.range([-48, 48], {value: -23, step: 1, label: "Aggregation-window offset h (hours)"}));
```

```js
const idx = windowCurves.h.indexOf(h);
const rB = windowCurves.bmkg[idx];
const rC = windowCurves.cpc[idx];
const rR = windowCurves.cpc_relabelled[idx];
```

<div class="grid grid-cols-3">
  <div class="card">
    <h2 style="color:#2980b9">vs BMKG (validation)</h2>
    <span class="big" style="color:#2980b9">${fmt(rB)}</span>
    at h = ${h} h
  </div>
  <div class="card">
    <h2 style="color:#c0392b">vs CPC-UNI (UTC label)</h2>
    <span class="big" style="color:#c0392b">${fmt(rC)}</span>
    the calibration reference
  </div>
  <div class="card">
    <h2 style="color:#e08e0b">vs CPC-UNI (relabelled +1 day)</h2>
    <span class="big" style="color:#e08e0b">${fmt(rR)}</span>
    harmonised to the local day
  </div>
</div>

```js
const long = windowCurves.h.flatMap((hh, i) => [
  {h: hh, r: windowCurves.cpc[i], series: "vs CPC-UNI (UTC label)"},
  {h: hh, r: windowCurves.bmkg[i], series: "vs BMKG (validation)"},
  {h: hh, r: windowCurves.cpc_relabelled[i], series: "vs CPC-UNI (relabelled +1 day)"}
]);
const marker = [
  {h, r: rB, series: "vs BMKG (validation)"},
  {h, r: rC, series: "vs CPC-UNI (UTC label)"},
  {h, r: rR, series: "vs CPC-UNI (relabelled +1 day)"}
];
```

```js
display(Plot.plot({
  width,
  height: 360,
  x: {label: "aggregation-window offset h (hours) →", domain: [-48, 48], grid: true},
  y: {label: "↑ daily Pearson r (GPM era, 2015-2021)", domain: [0, 0.7], grid: true},
  color: {
    legend: true,
    domain: Object.keys(WINDOW_COLORS),
    range: Object.values(WINDOW_COLORS)
  },
  marks: [
    Plot.ruleX([h], {stroke: "#333", strokeWidth: 1.4}),
    Plot.line(long, {x: "h", y: "r", stroke: "series", strokeWidth: 2}),
    Plot.dot(marker, {x: "h", y: "r", fill: "series", r: 5, stroke: "white", strokeWidth: 1}),
    Plot.crosshairX(long, {x: "h", y: "r", color: "series"}),
    Plot.text([{h, t: `h = ${h} h`}], {x: "h", y: 0.66, text: "t", fontWeight: "bold", dx: 4, textAnchor: "start"})
  ]
}));
```

<div class="dial-note">

**What the dial shows.** At **h = 0** (the UTC day) IMERG-L agrees best with **CPC-UNI** but poorly with **BMKG**; at **h = −23 h** (the local day) the picture flips, and BMKG recovers to about **0.57**. No single calendar label maximises agreement with both references at once, so a corrected product is best distributed on an explicit local-observation day with all references harmonised to it. The residual after harmonisation is the genuine retrieval-timing limit - it cannot be relabelled away.

</div>

## Daily vs monthly

The window fix matters **only at the daily scale**. Aggregate the same product to calendar months and the shift vanishes - a one-day window change just moves a day or two across each month boundary. Pooled over the GPM era (n = ${wg.daily_monthly.n_station_months.toLocaleString("en")} station-months):

```js
const dm = wg.daily_monthly;
display(html`<table style="border-collapse:collapse; font-size:14px; margin:0.6rem 0">
  <thead><tr>
    <th style="text-align:left; padding:4px 16px; border-bottom:2px solid var(--theme-foreground-muted)">Aggregation window</th>
    <th style="text-align:right; padding:4px 16px; border-bottom:2px solid var(--theme-foreground-muted)">Daily r</th>
    <th style="text-align:right; padding:4px 16px; border-bottom:2px solid var(--theme-foreground-muted)">Monthly r</th>
  </tr></thead>
  <tbody>${dm.rows.map((r) => html`<tr>
    <td style="padding:4px 16px; border-bottom:1px solid var(--theme-foreground-faintest)">${r.window}</td>
    <td style="text-align:right; padding:4px 16px; border-bottom:1px solid var(--theme-foreground-faintest)"><b>${r.daily.toFixed(2)}</b></td>
    <td style="text-align:right; padding:4px 16px; border-bottom:1px solid var(--theme-foreground-faintest)">${r.monthly.toFixed(2)}</td>
  </tr>`)}</tbody>
</table>`);
```

<div class="keyfinding">
<span class="kf-label">Finding</span>
The same product scores <b>r = 0.20</b> daily but <b>0.80</b> monthly at the UTC-day window. Re-windowing to the local day lifts the daily <i>r</i> to <b>0.57</b> but barely moves the monthly value (<b>0.80 → 0.81</b>). The daily ceiling lives in the day-by-day <i>pairing</i> - the timing - not in how much rainfall the product captures.
</div>

<div class="note" style="margin-top: 1.5rem">
The evidence behind this recovery - when it emerged (the satellite-era transition), that it holds in every timezone band and season, station by station, and across the whole 0.5&deg; domain - is on <a href="./window-detail">Whole-domain &amp; era</a>.
</div>
