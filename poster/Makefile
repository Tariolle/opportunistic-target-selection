PDFLATEX = pdflatex
FLAGS = -interaction=nonstopmode -halt-on-error
SOURCE_DATE_EPOCH ?= 0
export SOURCE_DATE_EPOCH
export FORCE_SOURCE_DATE = 1

POSTER_SOURCES = poster_beamer.tex \
	school_logo.png \
	logo_LITIS_cropped.png \
	arxiv_qr.png \
	../results/figures/standard/fig_headline_bars.pdf \
	../results/figures/standard/fig_lockin.pdf \
	../results/figures_theta/fig_theta.pdf

.PHONY: all clean

all: poster_beamer.pdf

poster_beamer.pdf: $(POSTER_SOURCES)
	$(PDFLATEX) $(FLAGS) poster_beamer.tex
	$(PDFLATEX) $(FLAGS) poster_beamer.tex

clean:
	del /Q *.aux *.log *.nav *.out *.snm *.toc *.vrb 2>NUL
