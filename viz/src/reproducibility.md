---
title: Reproducibility
---

<style>
.big { font-size: 1.9rem; font-weight: 700; display: block; line-height: 1.1; }
.linkcard { display: block; padding: 0.9rem 1.1rem; border: 1px solid var(--theme-foreground-faintest); border-radius: 6px; text-decoration: none; }
.linkcard:hover { border-color: var(--theme-foreground-focus); }
.linkcard .k { font-size: 12px; color: var(--theme-foreground-muted); text-transform: uppercase; letter-spacing: 0.04em; }
.linkcard .v { font-weight: 700; }
</style>

# Reproducible on free infrastructure

The whole LSEQM+DL pipeline re-runs end-to-end on a **free Google Colab CPU** - no GPU, no paid tier. The Bali subdomain, all 36 dekads of the 2001-2025 record, completes in **72.1 minutes**, well inside one free-tier session.

```js
const repro = await FileAttachment("data/repro.json").json();
const rt = repro.runtime;
```

<div class="grid grid-cols-3">
  <div class="card"><h2>End-to-end runtime</h2><span class="big">${rt.total}</span> min · Bali, all 36 dekads</div>
  <div class="card"><h2>Hardware</h2>${rt.hardware}</div>
  <div class="card"><h2>Coverage</h2>${rt.domain}</div>
</div>

## Where the 72 minutes go

```js
const SHORT = {
  "02": "02 · Bias correction + CNN", "03": "03 · Verification metrics",
  "04": "04 · Quality assessment", "05": "05 · Station validation", "06": "06 · Visualisation"
};
const rtData = rt.rows.map((r) => ({...r, lab: SHORT[r.nb]}));
```

```js
display(Plot.plot({
  width: width,
  height: 240,
  marginLeft: 170,
  x: {label: "batch runtime (min) →", grid: true, domain: [0, 52]},
  y: {label: null, domain: rtData.map((d) => d.lab)},
  marks: [
    Plot.barX(rtData, {x: "min", y: "lab", fill: "#2f7d9e", tip: true, channels: {notebook: "lab", "runtime (min)": "min"}}),
    Plot.text(rtData, {x: "min", y: "lab", text: (d) => `${d.min} min`, dx: 6,
      textAnchor: "start", fontSize: 11, fill: "var(--theme-foreground-muted)"}),
    Plot.ruleX([0])
  ]
}));
```

Verification (notebook 03) dominates the budget: it computes the full 31-metric catalogue at every pixel and every dekad. The per-dekad CNN training (notebook 02) is second. Environment: ${rt.env}.

Which of these outputs is worth trusting depends on the application - see [5.3 Application classes](./applications).

## Open code and data

```js
const L = repro.links;
```

<div class="grid grid-cols-2">
  ${html`<a class="linkcard" href=${L.github} target="_blank">
    <div class="k">Code · GitHub</div>
    <div class="v">bennyistanto/hybrid-bias-correction</div>
    <div style="font-size:13px;color:var(--theme-foreground-muted)">Modular src, six numbered notebooks, Colab mirror, documentation.</div>
  </a>`}
  ${html`<a class="linkcard" href=${L.zenodo} target="_blank">
    <div class="k">Data · Zenodo</div>
    <div class="v">doi:${L.zenodo_doi}</div>
    <div style="font-size:13px;color:var(--theme-foreground-muted)">Archived inputs and corrected outputs with a citable DOI.</div>
  </a>`}
</div>
