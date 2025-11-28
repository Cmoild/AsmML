import numpy as np
import time

size = 1000

a = np.random.randn(size, size).astype(np.float32)
b = np.random.randn(size, size).astype(np.float32)

start = time.time()

_ = a @ b

end = time.time()

print(end - start)
