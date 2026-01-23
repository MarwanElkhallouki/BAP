;; -*- lexical-binding: t; -*-

(TeX-add-style-hook
 "proposal-content"
 (lambda ()
   (LaTeX-add-labels
    "sec:introduction"
    "sec:literature"
    "sec:methodology"
    "sec:expected_results"))
 :latex)

