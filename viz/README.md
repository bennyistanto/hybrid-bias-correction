# Interactive results dashboard

An interactive companion to the hybrid bias-correction framework (LSEQM+DL) for daily
satellite precipitation over Indonesia. It walks through what the correction fixes, what it
structurally cannot, and the calendar-window artefact behind the daily-timing ceiling.

**Live:** <https://bennyistanto.github.io/hybrid-bias-correction/viz/>

Built with [Observable Framework](https://observablehq.com/framework/) as a static,
self-contained app. It reads only the pre-extracted JSON and PNG files committed under
`src/data/`, so it builds and deploys with no Python step.

## Layout

```
viz/
├── observablehq.config.js   # site config: sidebar groups, theme, base path
├── package.json             # Observable Framework, build scripts
├── src/
│   ├── index.md             # Start here
│   ├── *.md                 # one page per finding (staged skill, ceiling, window, ...)
│   ├── components/util.js   # shared constants and formatting helpers
│   └── data/                # pre-extracted JSON + maps/ PNGs (committed artefacts)
└── extract/                 # Python that regenerates src/data from pipeline outputs
```

## Local development

Requires Node.js 20+.

```bash
cd viz
npm install
npm run dev      # preview at http://localhost:3000
npm run build    # static build into dist/
```

## Regenerating the data

The pages never compute results in the browser; every number comes from a committed file in
`src/data/`. Those files are produced by the scripts in `extract/`, which read the framework's
own outputs and reuse the thesis figure helpers so the dashboard cannot drift from the paper:

- `extract_core.py` - headline stats, staged scorecards, Taylor and monthly-Taylor, seasonal cycle.
- `extract_maps.py` - per-stage QA and station-density map tiles (PNG).
- `extract_window.py` - calendar-window diagnostics and the gridded whole-domain maps.

Run them with the project's `climate` conda environment. They read from `data/output/` (the full
pipeline outputs archived on Zenodo, not shipped in the repo), so regeneration requires that data
in place. The committed `src/data/` artefacts let anyone build the dashboard without it.

## Deployment

`.github/workflows/quarto-publish.yml` builds this app on every push to `main`
(`npm ci && npm run build`) and copies `dist/` into the Quarto site at `/viz`, so the docs site
and the dashboard share one GitHub Pages deployment. The base path in `observablehq.config.js`
switches to `/hybrid-bias-correction/viz/` under CI and stays at `/` for local `npm run dev`.
