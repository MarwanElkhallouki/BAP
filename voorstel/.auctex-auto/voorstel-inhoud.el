;; -*- lexical-binding: t; -*-

(TeX-add-style-hook
 "voorstel-inhoud"
 (lambda ()
   (LaTeX-add-labels
    "sec:inleiding"
    "sec:literatuurstudie"
    "sec:methodologie"
    "sec:verwachte_resultaten"))
 :latex)

