---
title: Raising the ceiling
---

<style>
.card h2 { margin-top: 0; }
.kv { font-size: 13px; margin: 0.3rem 0; }
.kv b { color: var(--theme-foreground-muted); font-weight: 600; }
.badge { display: inline-block; background: #fdeede; color: #b5651d; font-weight: 700; font-size: 11px; padding: 1px 8px; border-radius: 999px; margin-left: 8px; }
.lift { font-size: 1.4rem; font-weight: 700; color: #b2182b; }
</style>

# Four paths beyond r ≈ 0.34

The marginal correction reaches its timing ceiling by design. Raising it means adding timing information the satellite did not carry - a different class of method. The thesis surveys four, positioned here by how far each sits from the current pipeline against how much ceiling lift it can be expected to deliver. One sits alone in the cheap-and-high-payoff corner - and it is the one this dashboard already quantifies.

```js
const pathsData = FileAttachment("data/paths.json").json();
```

```js
const paths = pathsData.paths;
const rec = pathsData.recommended;
const sel = view(Inputs.select(paths, {label: "Inspect a path", format: (p) => p.name, value: paths.find((p) => p.key === rec)}));
```

```js
const DIST = {1: "Small", 2: "Moderate", 3: "Large"};
const PAY = {1: "Low / open", 2: "Moderate", 3: "High (known)"};
display(Plot.plot({
  width: width,
  height: 390,
  marginLeft: 100,
  marginBottom: 50,
  marginTop: 26,
  x: {domain: [0.5, 3.5], ticks: [1, 2, 3], tickFormat: (d) => DIST[d], grid: true, label: "methodological distance from LSEQM+DL →"},
  y: {domain: [0.5, 3.5], ticks: [1, 2, 3], tickFormat: (d) => PAY[d], grid: true, label: "↑ expected ceiling lift"},
  marks: [
    Plot.dot(paths, {x: "distance_ord", y: "payoff", r: 9, stroke: "white", strokeWidth: 1,
      fill: (d) => d.key === sel.key ? "#b2182b" : "#2f7d9e",
      tip: true, channels: {Path: "name", distance: "distance", cost: "cost"}}),
    Plot.text(paths, {x: "distance_ord", y: "payoff", text: "name", dy: -15, fontSize: 11,
      fontWeight: (d) => d.key === sel.key ? 700 : 400, lineWidth: 10})
  ]
}));
```

```js
display(html`<div class="card">
  <h2>${sel.name}${sel.key === rec ? html`<span class="badge">most actionable</span>` : ""}</h2>
  <p class="kv"><b>Methodological distance:</b> ${sel.distance}</p>
  <p class="kv"><b>Infrastructure cost:</b> ${sel.cost}</p>
  <p class="kv"><b>Ceiling effect:</b> ${sel.effect}</p>
  ${sel.lift ? html`<p class="kv"><b>Expected lift:</b> <span class="lift">${sel.lift[0]} → ${sel.lift[1]}</span> in daily Pearson r</p>` : ""}
  <p style="margin:0.6rem 0 0">${sel.summary}</p>
</div>`);
```

## The low-hanging fruit is already on the dashboard

Three of the four paths are moderate-to-large departures with uncertain or partial payoff. The fourth - **local-day re-aggregation** - is a *small* change (re-window the input before the unchanged pipeline) with the only **quantified** lift: **r 0.34 → 0.57**, the recovery you can slide through yourself on the [calendar-window page](./window). It is the path the reproducibility contribution most directly enables: the daily-aggregation window is an exposed pipeline parameter, so any user can re-aggregate the archive and re-verify against the same BMKG set.

<div class="note" style="margin-top:1.5rem">
All four paths leave the LSEQM+DL pipeline intact for the [well-served applications](./applications) - users who need the distribution, not the calendar day, already have their product. These routes matter only for the timing-critical class, and the cheapest of them targets exactly the [convention artefact](./window) this work diagnosed.
</div>
