# %%
import itertools as it
target_levels = ["classical", "country", "rock", "pop", "hiphop", "jazz", "blues", "disco", "metal", "reggae"]

target_pairs = [target_pair for target_pair in it.combinations(target_levels, 2)]
# %%
