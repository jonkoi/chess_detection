import os
import glob
from main import main_func
import numpy as np

open('fail.txt', 'w').close()
a = glob.glob('test/*.jpg')
for path in a:
    print(path)
    main_func(path, test=True)
# a = np.array([1,2,3])
