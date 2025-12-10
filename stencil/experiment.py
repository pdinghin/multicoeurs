#!/usr/bin/env python

import subprocess

tests = [
    ["./stencil", "--nb-iterations 10","--mesh-width 400","--mesh-height 400"],
    ["./stencil", "--nb-iterations 10","--mesh-width 1000","--mesh-height 1000"],
    ["./stencil", "--nb-iterations 10","--mesh-width 2000","--mesh-height 2000"],
    ["./stencil", "--nb-iterations 10","--mesh-width 4000","--mesh-height 4000"],
    ["./stencil_vec", "--nb-iterations 10","--mesh-width 400","--mesh-height 400"],
    ["./stencil_vec", "--nb-iterations 10","--mesh-width 1000","--mesh-height 1000"],
    ["./stencil_vec", "--nb-iterations 10","--mesh-width 2000","--mesh-height 2000"],
    ["./stencil_vec", "--nb-iterations 10","--mesh-width 4000","--mesh-height 4000"],
    ["./stencil_starpu", "--nb-iterations 10","--mesh-width 400","--mesh-height 400"],
    ["./stencil_starpu", "--nb-iterations 10","--mesh-width 1000","--mesh-height 1000"],
    ["./stencil_starpu", "--nb-iterations 10","--mesh-width 2000","--mesh-height 2000"],
    ["./stencil_starpu", "--nb-iterations 10","--mesh-width 4000","--mesh-height 4000"],
    ["./stencil_starpu_vec", "--nb-iterations 10","--mesh-width 400","--mesh-height 400"],
    ["./stencil_starpu_vec", "--nb-iterations 10","--mesh-width 1000","--mesh-height 1000"],
    ["./stencil_starpu_vec", "--nb-iterations 10","--mesh-width 2000","--mesh-height 2000"],
    ["./stencil_starpu_vec", "--nb-iterations 10","--mesh-width 4000","--mesh-height 4000"],
]

for test in tests:
    nom = test[0]
    subprocess.run(
        test,              
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    print(f"✅ {nom} terminé.")
print("\nTous les tests sont terminés.")