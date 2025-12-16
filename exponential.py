import math
from sys import argv


def my_exp(x: float) -> float:
    y = x * math.log2(math.e)
    n = math.floor(y)
    f = y - n

    a0 = 1.0
    a1 = 0.6931471805599453
    a2 = 0.2402265069591007
    a3 = 0.05550410866482158
    a4 = 0.009618129107628477
    a5 = 0.001333355814642844

    two_f = a0 + a1 * f + a2 * f**2 + a3 * f**3 + a4 * f**4 + a5 * f**5
    ret = two_f * 2**n
    return ret


print(argv)
mexp = math.exp(float(argv[1]))
myexp = my_exp(float(argv[1]))
print(f"Math.exp: {mexp}")
print(f"My exp: {myexp}")
print(f"Difference: {abs(mexp - myexp)}")
