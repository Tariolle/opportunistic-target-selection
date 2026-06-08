# CAp 2026 Poster

This directory contains the A0 portrait poster for the CAp 2026 presentation:

- `poster_beamer.tex`: conventional `beamerposter` layout.

Compile it from this directory:

```bash
pdflatex -interaction=nonstopmode -halt-on-error poster_beamer.tex
```

Or with `make`:

```bash
make
```

The poster reuses figures from `../results/`.
