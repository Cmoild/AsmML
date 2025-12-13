import numpy as np

size = 256
TILE_HEIGHT = 8
TILE_WIDTH = 8

TILE_M = 64  # Для L1 кэша
TILE_N = 256  # Для повторного использования B в L2
TILE_K = 32  # Для удержания в регистрах

assert size % TILE_HEIGHT == 0 and size % TILE_WIDTH == 0

a = np.random.randn(size, size).astype(np.float32)
b = np.random.randn(size, size).astype(np.float32)


def matmul(a: np.ndarray, b: np.ndarray):
    M, K = a.shape
    _, N = b.shape
    result = np.zeros((a.shape[0], b.shape[1]), dtype=np.float32)
    for rcol in range(0, N, TILE_N):
        for lrow in range(0, M, TILE_M):
            acc = np.zeros((TILE_M, TILE_N))  # Аккумулятор
            for lcol in range(0, K, TILE_K):
                tile_a = a[lrow : lrow + TILE_M, lcol : lcol + TILE_K]
                tile_b = b[lcol : lcol + TILE_K, rcol : rcol + TILE_N]
                acc += tile_a @ tile_b  # Аккумуляция в быстрой памяти
            result[lrow : lrow + TILE_M, rcol : rcol + TILE_N] = acc
    return result


assert np.allclose(matmul(a, b), a @ b, atol=1e-2)
# print(matmul(a, b))
# print(a @ b)
