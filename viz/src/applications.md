---
title: Who it serves
---

<style>
.verdict { display: inline-block; padding: 3px 12px; border-radius: 999px; font-weight: 700; font-size: 0.95rem; }
.verdict.good { background: #e7f5e7; color: #1a7e2e; }
.verdict.bad { background: #fdeede; color: #b5651d; }
.applist { list-style: none; padding: 0; margin: 0.5rem 0; }
.applist li { padding: 4px 0; border-bottom: 1px solid var(--theme-foreground-faintest); font-size: 14px; }
.topic { display: inline-block; font-size: 11px; color: var(--theme-foreground-muted); border: 1px solid var(--theme-foreground-faintest); border-radius: 999px; padding: 0 7px; margin-left: 6px; }
</style>

# What it is good for - and what it is not

The corrected product moves the daily **distribution** to the gauge but inherits raw satellite **timing** (the [r ≈ 0.35 ceiling](./ceiling), measured against CPC-UNI at daily resolution, native window, per-pixel spatial median then averaged over the 36 dekads). So the useful question is not "is it accurate?" but "does your application need the distribution or the calendar day?" Pick one:

```js
const repro = await FileAttachment("data/repro.json").json();
```

```js
const app = view(Inputs.select(repro.applications, {
  label: "Your application",
  format: (d) => d.name,
  sort: (a, b) => d3.ascending(a.name, b.name)
}));
```

```js
const sm = repro.served[String(app.served)];
display(html`<div class="card">
  <span class="verdict ${app.served ? "good" : "bad"}">${sm.label}</span>
  <p style="margin:0.7rem 0 0.3rem"><b>${app.name}</b> depends on <b>${sm.skill.toLowerCase()}</b>${app.served ? ", which the correction moves to the gauge target - to within 1-5% against the independent BMKG stations (bias −0.6%, SDR 1.00, Q99 1.01). Against CPC-UNI, the in-sample target it was fitted to, the upper tail overshoots instead, by 15-20% (SDR 1.15, Q99 1.20)." : ", which marginal correction barely moves - against CPC-UNI the daily Pearson r stays pinned close to its raw value, holding near 0.35 across all three stages (0.343 / 0.345 / 0.348)."}</p>
  <p style="margin:0; font-size:13px; color:var(--theme-foreground-muted)">Judge it by: ${sm.metrics}</p>
</div>`);
```

```js
const served = repro.applications.filter((a) => a.served);
const poorly = repro.applications.filter((a) => !a.served);
const appList = (arr) => html`<ul class="applist">${arr.map((a) =>
  html`<li>${a.name}<span class="topic">${a.topic}</span></li>`)}</ul>`;
```

<div class="grid grid-cols-2">
  <div>
    <span class="verdict good">Well-served</span> - needs the daily distribution
    ${appList(served)}
  </div>
  <div>
    <span class="verdict bad">Poorly-served</span> - needs day-specific timing
    ${appList(poorly)}
  </div>
</div>

The split tracks day-specific **timing, not topic**: flood, hydrology and agriculture each appear on **both** sides. A 25-year flood-frequency analysis is well-served; real-time flood nowcasting on a named date is not - the same field, judged on different axes.
