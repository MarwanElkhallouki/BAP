# Paper Humanizer — Agent Memory

## Project context
- Bachelor's thesis (HOGENT), English, AI & Data Engineering
- Author: Marwan Elkhallouki
- Topic: XAI + drift detection for industrial visual inspection (ResNet-50, MVTec AD, ImageNet-C)
- LaTeX compiler: XeLaTeX; citations use \autocite{} and \textcite{} (biblatex/biber)

## Humanization preferences confirmed by user
- Tone: "sharp, knowledgeable engineer" — not a committee, not a textbook
- Short punchy sentences mixed with longer ones; deliberate variation
- Active voice preferred; passive allowed occasionally
- No: "furthermore", "moreover", "it is worth noting", "it is important to", "in conclusion"
- No: "in practice", "a deliberate choice", "arguably", "at its core", "ultimately", "the obvious limitation"
- No em dashes (---) anywhere in prose: replace with commas, colons, semicolons, or parentheses
- Subsection headings that used "Subsection --- Name" pattern replaced with "Subsection: Name" colon style
- No uniform paragraph length — vary deliberately
- Preserve all: \autocite{}, \textcite{}, \begin{quote}...\end{quote}, LaTeX environments, section labels, figure/table environments
- Do NOT change any numbers, algorithm descriptions, dataset statistics, or technical claims

## Workflow that worked well
1. Read all files in parallel before touching anything
2. Identify the flattest / most formulaic passages (usually chapter intros and transition paragraphs)
3. Write full rewrites rather than small edits when a section has systemic AI patterns
4. Verify the \begin{quote} block in inleiding.tex is untouched after rewrite
5. Final read pass to confirm LaTeX structure and citations intact

## Common AI patterns found in this thesis draft
- Uniform paragraph size (every paragraph ~4 sentences)
- Transition openers: "The aim throughout is...", "This chapter surveys..."
- Redundant framing: "The combination is intentional." followed by an over-explained rationale
- Passive chains: "is described", "is used", "is computed", "is applied" in consecutive sentences
- Closing sentences that restate the section title rather than adding anything
