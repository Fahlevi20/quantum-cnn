# %%
import itertools as it
target_levels = ["classical", "country", "rock", "pop", "hiphop", "jazz", "blues", "disco", "metal", "reggae"]

target_pairs = [target_pair for target_pair in it.combinations(target_levels, 2)]
# %%
# Quicksort
def f(x):
    if len(x)<=1:
        return x
    else:
        y = [a for a in x if a<x[0]]
        z = [a for a in x if a>x[0]]
        return f(y)+[x[0]]+f(z)
# %%
f([2,1,3, 10, 15])
# %%
