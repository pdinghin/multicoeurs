#!/usr/bin/env python

import os

tests = [
    "./stencil_starpu --nb-iterations 10 --mesh-width 400 --mesh-height 400",
    "./stencil_starpu --nb-iterations 10 --mesh-width 1000 --mesh-height 1000",
    "./stencil_starpu --nb-iterations 10 --mesh-width 2000 --mesh-height 2000",
    "./stencil_starpu --nb-iterations 10 --mesh-width 4000 --mesh-height 4000",
    "./stencil_starpu --nb-iterations 10 --mesh-width 200 --mesh-height 200",
    "./stencil_starpu --nb-iterations 10 --mesh-width 1500 --mesh-height 1500",
    "./stencil_starpu --nb-iterations 10 --mesh-width 3000 --mesh-height 3000",
    "./stencil_starpu --nb-iterations 10 --mesh-width 3500 --mesh-height 3500",
]

for test in tests:
    os.system(test)
    