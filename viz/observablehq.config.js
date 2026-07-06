// Observable Framework config. Static build -> GitHub Pages.
// In CI the dashboard is published as a subpath of the Quarto Pages site
// (bennyistanto.github.io/hybrid-bias-correction/viz/); locally it stays at
// the root so `npm run dev` serves from localhost:3000/.
export default {
  title: "Hybrid Bias Correction",
  root: "src",
  base: process.env.GITHUB_ACTIONS ? "/hybrid-bias-correction/viz/" : "/",
  theme: ["air", "wide"],
  toc: false,
  // Shared component styles, global to every page (Key finding callout,
  // stage-attribution cards). Keeps the look consistent without per-page CSS.
  head: `<style>
    :root { --observablehq-max-width: 1760px; }
    #observablehq-footer { display: none; }
    .keyfinding { border-left: 3px solid #b2182b; background: rgba(178,33,43,0.06); padding: 0.55rem 1rem; border-radius: 0 4px 4px 0; margin: 1.2rem 0; }
    .keyfinding .kf-label { display: block; font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: #b2182b; font-weight: 700; margin-bottom: 2px; }
    .finding-card { border: 1px solid var(--theme-foreground-faintest); border-radius: 6px; padding: 0.85rem 1rem; }
    .finding-card h3 { margin: 0 0 0.4rem; font-size: 1rem; }
    .stage-tag { display: inline-block; font-size: 11px; font-weight: 700; padding: 1px 9px; border-radius: 999px; color: white; }
    blockquote b, .keyfinding b, .finding-card b, .note b { font-weight: 700; }
  </style>`,
  pages: [
    {name: "Start here", path: "/"},
    {
      name: "The correction",
      open: true,
      pages: [
        {name: "What it fixes", path: "/staged-skill"},
        {name: "Detection by threshold", path: "/skill"},
        {name: "Spatial quality", path: "/spatial"},
        {name: "Stations & seasons", path: "/stations"},
      ],
    },
    {
      name: "The ceiling - and the fix",
      open: true,
      pages: [
        {name: "The timing ceiling", path: "/ceiling"},
        {name: "The calendar window", path: "/window"},
        {name: "Whole-domain & era", path: "/window-detail"},
      ],
    },
    {
      name: "Trusting & using it",
      open: true,
      pages: [
        {name: "Sensitivity", path: "/sensitivity"},
        {name: "Station-density mask", path: "/density"},
        {name: "Who it serves", path: "/applications"},
        {name: "Reproducibility", path: "/reproducibility"},
        {name: "Raising the ceiling", path: "/paths"},
      ],
    },
  ],
  footer: "",
};
