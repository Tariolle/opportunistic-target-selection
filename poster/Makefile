PDFLATEX = pdflatex
FLAGS = -interaction=nonstopmode -halt-on-error

.PHONY: all clean

all: poster_beamer.pdf poster_gemini.pdf

poster_beamer.pdf: poster_beamer.tex
	$(PDFLATEX) $(FLAGS) poster_beamer.tex
	$(PDFLATEX) $(FLAGS) poster_beamer.tex

poster_gemini.pdf: poster_gemini.tex beamerthemegemini.sty
	$(PDFLATEX) $(FLAGS) poster_gemini.tex
	$(PDFLATEX) $(FLAGS) poster_gemini.tex

clean:
	del /Q *.aux *.log *.nav *.out *.snm *.toc *.vrb 2>NUL
