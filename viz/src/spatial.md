---
title: Spatial quality
---

<style>
.legendbar { height: 12px; border-radius: 3px; border: 1px solid var(--theme-foreground-faintest); }
.legendrow { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--theme-foreground-muted); max-width: 420px; }
.chip { display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 13px; border: 1px solid var(--theme-foreground-faintest); }
.chip.on { background: var(--theme-foreground); color: var(--theme-background); font-weight: 700; }
.maptile { width: 100%; max-width: 960px; display: block; border: 1px solid var(--theme-foreground-faintest); border-radius: 4px; }
.swatch { display: inline-block; width: 13px; height: 13px; border-radius: 2px; vertical-align: -1px; }
</style>

# Spatial quality

Per-pixel QA **climatology** - the annual mean over the 36 dekads of the gauge-referenced quality grid (`qualitysd_cpc`, 0.1&deg;, **19,393** land pixels). Pick a dimension and toggle the stage: the colour scale is fixed across stages, so any change you see between LS, LSEQM and LSEQM+DL is real, not rescaled.

```js
const mapMeta = FileAttachment("data/map_meta.json").json();
```

```js
// Framework needs static FileAttachment literals, so every tile is declared here.
const TILE = {
  cqi_ls: FileAttachment("data/maps/cqi_ls.png"), cqi_lseqm: FileAttachment("data/maps/cqi_lseqm.png"), cqi_lseqmdl: FileAttachment("data/maps/cqi_lseqmdl.png"), cqi_diff: FileAttachment("data/maps/cqi_diff.png"),
  confidence_ls: FileAttachment("data/maps/confidence_ls.png"), confidence_lseqm: FileAttachment("data/maps/confidence_lseqm.png"), confidence_lseqmdl: FileAttachment("data/maps/confidence_lseqmdl.png"), confidence_diff: FileAttachment("data/maps/confidence_diff.png"),
  categorical_ls: FileAttachment("data/maps/categorical_ls.png"), categorical_lseqm: FileAttachment("data/maps/categorical_lseqm.png"), categorical_lseqmdl: FileAttachment("data/maps/categorical_lseqmdl.png"), categorical_diff: FileAttachment("data/maps/categorical_diff.png"),
  basic_ls: FileAttachment("data/maps/basic_ls.png"), basic_lseqm: FileAttachment("data/maps/basic_lseqm.png"), basic_lseqmdl: FileAttachment("data/maps/basic_lseqmdl.png"), basic_diff: FileAttachment("data/maps/basic_diff.png"),
  distribution_ls: FileAttachment("data/maps/distribution_ls.png"), distribution_lseqm: FileAttachment("data/maps/distribution_lseqm.png"), distribution_lseqmdl: FileAttachment("data/maps/distribution_lseqmdl.png"), distribution_diff: FileAttachment("data/maps/distribution_diff.png"),
  temporal_ls: FileAttachment("data/maps/temporal_ls.png"), temporal_lseqm: FileAttachment("data/maps/temporal_lseqm.png"), temporal_lseqmdl: FileAttachment("data/maps/temporal_lseqmdl.png"), temporal_diff: FileAttachment("data/maps/temporal_diff.png"),
};
const STAGE_FILE = {"LS": "ls", "LSEQM": "lseqm", "LSEQM+DL": "lseqmdl"};
const DIM_NAMES = new Map([
  ["Continuous Quality Index (CQI)", "cqi"], ["Confidence level", "confidence"],
  ["Categorical class", "categorical"], ["Basic statistical", "basic"],
  ["Distribution", "distribution"], ["Temporal", "temporal"]
]);
```

```js
const dimKey = view(Inputs.select(DIM_NAMES, {label: "Dimension", value: "cqi"}));
```

```js
const stage = view(Inputs.radio(["LS", "LSEQM", "LSEQM+DL"], {value: "LSEQM+DL", label: "Stage"}));
```

```js
const dim = mapMeta.dims[dimKey];
const means = mapMeta.means[dimKey];
const mapSrc = await TILE[`${dimKey}_${STAGE_FILE[stage]}`].url();
const diffSrc = await TILE[`${dimKey}_diff`].url();
```

<div class="grid grid-cols-2">
  <div>
    <h3>${dim.label} - ${stage}</h3>
    ${html`<img class="maptile" src=${mapSrc} alt=${`${dim.label} ${stage}`}>`}
    ${seqLegend(dim)}
    <p style="font-size:13px; margin-top:0.8rem">
      Domain mean by stage:
      ${["LS", "LSEQM", "LSEQM+DL"].map((s) => html`<span class="chip ${s === stage ? "on" : ""}">${s} ${means[STAGE_FILE[s]]}</span> `)}
    </p>
  </div>
  <div>
    <h3>Where the pipeline moves it - LSEQM+DL − LS</h3>
    ${html`<img class="maptile" src=${diffSrc} alt="difference">`}
    ${diffLegend(dim)}
    <p style="font-size:13px; margin-top:0.8rem; color:var(--theme-foreground-muted)">
      Green = the corrected product raised this quality dimension; brown = lowered it. Most of Indonesia is near-neutral: the marginal correction reshapes the <b>distribution</b> without moving the per-pixel composite much.
    </p>
  </div>
</div>

```js
function seqLegend(dim) {
  if (dim.kind === "cat") {
    const labs = ["1 · poor", "2 · fair", "3 · good"];
    const cols = ["#d73027", "#fee08b", "#1a9850"];
    return html`<div class="legendrow" style="margin-top:6px">${labs.map((l, i) =>
      html`<span><span class="swatch" style=${`background:${cols[i]}`}></span> ${l}</span>`)}</div>`;
  }
  return html`<div>
    <div class="legendrow" style="margin-top:6px">
      <span>${dim.vmin}</span>
      <div class="legendbar" style="flex:1; background:linear-gradient(to right,#d73027,#f46d43,#fdae61,#fee08b,#d9ef8b,#a6d96a,#66bd63,#1a9850)"></div>
      <span>${dim.vmax}</span>
    </div>
    <div style="font-size:11px;color:var(--theme-foreground-muted)">low → high quality</div>
  </div>`;
}
function diffLegend(dim) {
  const a = dim.diff_abs;
  return html`<div class="legendrow" style="margin-top:6px">
    <span>−${a}</span>
    <div class="legendbar" style="flex:1; background:linear-gradient(to right,#8c510a,#d8b365,#f6e8c3,#f5f5f5,#c7eae5,#5ab4ac,#01665e)"></div>
    <span>+${a}</span></div>`;
}
```

<div class="note" style="margin-top:1.5rem">
This grid QA is measured <b>in-sample against CPC-UNI</b>; it is a different lens from the out-of-sample <a href="./staged-skill">BMKG scorecard</a>, where the distribution gains appear. The composite here is detection-weighted, so LS - which keeps more of IMERG-L's raw hit rate - reads marginally higher on CQI. That is the same designed trade-off the scorecard shows as a POD drop.<br><br>
Next: a station-density overlay to answer "why is quality lower here?", and the seasonal (dekad) cycle of CQI.
</div>
