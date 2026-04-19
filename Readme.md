# Project 8 -- Numerical Integration (Riemann)

**Course:** CST-305: Principles of Modeling and Simulation
**Author:** Jared Walker
**Semester:** Spring 2026
**University:** Grand Canyon University

---

## Overview

This project implements a numerical integration tool based on Riemann sums and applies it to two contexts. Part 1 evaluates Riemann sums for four different functions using left, right, and midpoint endpoint methods, with analytic derivations of the n→∞ limits. Part 2 uses the tool to compute the total data downloaded over a 30-minute experiment sampled at 1-minute intervals.

---

## Requirements

- Python 3.8+
- numpy
- scipy
- matplotlib

```bash
pip install numpy scipy matplotlib
```

---

## Running

```bash
python3 project8_riemann.py
```

Prints all Riemann sum values, derivations, and numerical verifications to the console. Saves 5 figures as PNG files in the current directory.

---

## Output Files

| File | Contents |
|------|----------|
| `proj8_part1a_left.png` | Left-hand Riemann rectangles, sin(x)+1 |
| `proj8_part1a_right.png` | Right-hand Riemann rectangles, sin(x)+1 |
| `proj8_part1a_midpoint.png` | Midpoint Riemann rectangles, sin(x)+1 |
| `proj8_part1c1_lnx.png` | ln(x) integral with n=100 rectangles |
| `proj8_part2_download.png` | 30-min download rate experiment |

---

## References

- Thomas, G. B. (2014). *Thomas' Calculus* (13th ed.). Pearson.
- SciPy Documentation: https://docs.scipy.org/doc/scipy/reference/integrate.html
