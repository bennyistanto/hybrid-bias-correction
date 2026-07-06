---
title: Sensitivity
---

<style>
.callout { border-left: 3px solid var(--theme-foreground-focus); padding: 0.4rem 0 0.4rem 1rem; margin: 1rem 0; }
.callout .big { font-size: 1.5rem; font-weight: 700; }
table.sweeptable { width: 100%; border-collapse: collapse; font-size: 13px; }
table.sweeptable th, table.sweeptable td { padding: 3px 8px; text-align: right; border-bottom: 1px solid var(--theme-foreground-faintest); }
table.sweeptable th:first-child, table.sweeptable td:first-child { text-align: left; }
table.sweeptable tr.group td { border-top: 2px solid var(--theme-foreground-muted); }
table.sweeptable tr.def td { background: #fdeede; font-weight: 700; }
</style>

# Sensitivity to the three configuration parameters

Three parameters were set by design judgement, not formal optimisation: the DL blend weight (**α = 0.70**), the GPD tail threshold (**80th percentile**), and the density saturation count (**2**). A Bali sweep varies each around its default while holding the others fixed. The result is both reassuring and a little humbling - the daily correlation barely moves.

```js
const sens = FileAttachment("data/sensitivity.json").json();
```

```js
const env = sens.envelope;
```

<div class="callout">
Across all <b>15</b> settings, pooled Pearson <span class="big">r</span> spans only <span class="big">${sens.span}</span> - the whole envelope is <b>[${env[0]}, ${env[1]}]</b>. Daily correlation is nearly invariant to every knob the pipeline exposes.
</div>

```js
function panel(p) {
  const pts = p.values.map((v, i) => ({x: v, r: p.r[i], rb: p.rb[i], sdr: p.sdr[i], rmse: p.rmse[i], nse: p.nse[i]}));
  const di = p.values.indexOf(p.default);
  const xmin = Math.min(...p.values), xmax = Math.max(...p.values);
  const pad = (xmax - xmin) * 0.08;
  return Plot.plot({
    width: 300, height: 250, marginBottom: 44, marginLeft: 48, marginTop: 24,
    x: {label: `${p.label} →`, domain: [xmin - pad, xmax + pad], ticks: p.values, tickFormat: (d) => `${d}`},
    y: {label: p.key === "alpha" ? "↑ Pearson r (pooled)" : null, domain: [0.30, 0.36], grid: true},
    marks: [
      Plot.rect([{}], {x1: xmin - pad, x2: xmax + pad, y1: env[0], y2: env[1], fill: "#1f78b4", fillOpacity: 0.10}),
      Plot.line(pts, {x: "x", y: "r", stroke: "#1f78b4", strokeWidth: 2}),
      Plot.dot(pts, {x: "x", y: "r", r: 4, fill: "#1f78b4", tip: true,
        channels: {[p.label]: "x", "Pearson r": "r", "RB %": "rb", SDR: "sdr", RMSE: "rmse", NSE: "nse"}}),
      Plot.dot([pts[di]], {x: "x", y: "r", symbol: "star", r: 9, fill: "#b2182b"}),
      Plot.text([pts[di]], {x: "x", y: "r", text: ["default"], dy: -13, fontSize: 10, fill: "#b2182b", fontWeight: 600})
    ]
  });
}
display(html`<div class="grid grid-cols-3">${sens.params.map(panel)}</div>`);
```

The shaded band is the full sweep envelope; every curve stays inside it, and each red star marks the operating default. **Hover any point** for its full metric set. There is no sharp optimum to miss - the response surface is flat.

## The full sweep

All fifteen settings, pooled per-pixel over the Bali subdomain. The operating default of each parameter is shaded. Pearson *r* holds in **[${env[0]}, ${env[1]}]**, but the distribution metrics do respond - the **GPD threshold** drives the most movement (SDR **0.923 → 0.867** across its range), which is why it is the one setting a downstream user would most need to revisit.

```js
display(html`<table class="sweeptable">
  <thead><tr><th>Parameter</th><th>Value</th><th>r</th><th>RB %</th><th>SDR</th><th>RMSE</th><th>NSE</th></tr></thead>
  <tbody>${sens.params.flatMap((p) => p.values.map((v, i) => {
    const cls = (i === 0 ? "group " : "") + (v === p.default ? "def" : "");
    return html`<tr class=${cls}>
      <td>${i === 0 ? p.label : ""}</td>
      <td>${v}${p.unit ? " " + p.unit : ""}</td>
      <td>${p.r[i].toFixed(3)}</td><td>${p.rb[i].toFixed(1)}</td><td>${p.sdr[i].toFixed(3)}</td>
      <td>${p.rmse[i].toFixed(2)}</td><td>${p.nse[i].toFixed(2)}</td>
    </tr>`;
  }))}</tbody>
</table>`);
```

## Why this is the expected result, not a lucky one

The parameters shape *how* the marginal correction is applied, but none of them add day-to-day timing information. Pearson *r* is bounded by the satellite retrieval (the [r ≈ 0.34 ceiling](./ceiling)), so tuning α, the GPD threshold, or the saturation count moves skill only within the noise. Formal optimisation would not lift the ceiling - that needs the [calendar-window fix](./window) or a different [class of method](./paths).

The one place a parameter matters more than this chart suggests is **spatial**: the [density saturation count](./density) barely moves pooled *r*, but it decides *where* the CNN is allowed to act. Invariance in the pooled number is not the same as invariance in the map. The thesis flags all three as reasonable-but-unoptimised settings, and this sweep is the evidence that the choice is safe within its range.
