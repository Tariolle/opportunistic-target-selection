# CAp 2026 Poster Drafts

Two A0 portrait poster drafts are provided for comparison:

- `poster_beamer.tex`: conventional `beamerposter` layout.
- `poster_gemini.tex`: lightweight local Gemini-style layout.

Compile both from this directory:

```bash
pdflatex -interaction=nonstopmode -halt-on-error poster_beamer.tex
pdflatex -interaction=nonstopmode -halt-on-error poster_gemini.tex
```

Or with `make`:

```bash
make
```

Both drafts reuse figures from `../results/`.
