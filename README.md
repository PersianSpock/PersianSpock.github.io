# PersianSpock.github.io

Source for my personal portfolio website.

**Live site:** https://persianspock.github.io

The site is a single static page (hand-written HTML and CSS, no framework or
build step) that introduces who I am, links to my CV, and presents one featured
machine-learning project as a case study.

## Repository structure

```
PersianSpock.github.io/
├── index.html                     # portfolio homepage (semantic HTML)
├── css/
│   └── style.css                  # single stylesheet, design tokens + responsive
├── assets/
│   └── favicon.svg                # site favicon
├── cv/
│   ├── cv.tex                     # CV — LaTeX source
│   └── cv.pdf                     # CV — compiled PDF (linked from the site)
├── projects/
│   └── churn-prediction/
│       └── README.md              # case study; links to source repo + Colab
├── report/
│   ├── report.tex                 # accompanying report — LaTeX source
│   └── report.pdf                 # accompanying report — compiled PDF
├── .gitignore
└── README.md                      # this file
```

## Featured project

**Telecom Customer Churn Prediction** — an end-to-end supervised-ML pipeline
(SMOTE for imbalance, an eight-model classifier bench plus a stacking ensemble,
and a Keras ANN) on the IBM Telco dataset. Case study in
[`projects/churn-prediction/`](projects/churn-prediction/README.md); canonical
code at [github.com/PersianSpock/Python-AI](https://github.com/PersianSpock/Python-AI).

## Deployment

Hosted on **GitHub Pages**. Because the repository is named
`PersianSpock.github.io`, GitHub serves it from the repository root at the
user-level domain — no `gh-pages` branch, no CI, no configuration beyond enabling
Pages. Pushing to `main` redeploys the site automatically.

To enable (one-time): **Settings → Pages → Source: `main`, folder `/ (root)`**.

## Local preview

It is a static site, so any static server works:

```bash
python3 -m http.server 8000   # then open http://localhost:8000
```

## Building the PDFs

Both the CV and the report are LaTeX:

```bash
pdflatex -output-directory=cv     cv/cv.tex
pdflatex -output-directory=report report/report.tex
```
