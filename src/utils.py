import numpy as np
import pandas as pd
import pickle

FEATURES = ['ndrift_signal','denoised_signal']

def get_feature_importance(model):
    print("Features sorted by their score:")
    print(sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), FEATURES),
             reverse=True))

rf = None
with open ('models/rFOCslow_0.pkl', 'rb') as f:
    rF = pickle.load(f)

print(FEATURES)
get_feature_importance(rF)
