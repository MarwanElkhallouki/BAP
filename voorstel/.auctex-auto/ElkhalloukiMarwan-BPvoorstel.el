;; -*- lexical-binding: t; -*-

(TeX-add-style-hook
 "ElkhalloukiMarwan-BPvoorstel"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("hogent-article" "")))
   (TeX-run-style-hooks
    "latex2e"
    "voorstel-inhoud"
    "hogent-article"
    "hogent-article10")
   (LaTeX-add-bibliographies
    "voorstel"))
 :latex)

