TEX = pdflatex -interaction=nonstopmode
BIB = bibtex
SRC = main

.PHONY: all clean

all: $(SRC).pdf

$(SRC).pdf: $(SRC).tex references.bib
	$(TEX) $(SRC)
	$(BIB) $(SRC)
	$(TEX) $(SRC)
	$(TEX) $(SRC)

clean:
	rm -f $(SRC).pdf $(SRC).aux $(SRC).bbl $(SRC).blg $(SRC).log $(SRC).out $(SRC).toc $(SRC).lof $(SRC).lot $(SRC).fls $(SRC).fdb_latexmk
