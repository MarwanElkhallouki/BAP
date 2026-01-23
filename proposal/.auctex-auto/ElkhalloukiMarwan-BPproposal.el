;; -*- lexical-binding: t; -*-

(TeX-add-style-hook
 "ElkhalloukiMarwan-BPproposal"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("hogent-article" "english")))
   (TeX-run-style-hooks
    "latex2e"
    "proposal-content"
    "hogent-article"
    "hogent-article10")
   (LaTeX-add-bibliographies
    "proposal"))
 :latex)

