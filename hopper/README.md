Instructions to reproduce the hopper results.

First, run

``
  python hopper.py
``

`hopper.py` will compute a trajectory using the baseline first and return an error. 


Then, in `hopper.py`, replace lines 25 and 26 with

```
  B_compute_solution_baseline = False
  B_compute_solution_saa = True
```

and rerun the script 

``
  python hopper.py
``

until it finishes and returns all results.