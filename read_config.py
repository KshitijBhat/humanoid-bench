import pickle
from ml_collections import ConfigDict

with open('/home/kshitij/humanoid-bench/results/sac_g1-window-v0_1774054741/config.pkl', 'rb') as f:
    config = pickle.load(f)

# You can then access it with dot notation if it's a ConfigDict
print(config)