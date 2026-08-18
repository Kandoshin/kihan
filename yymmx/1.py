def euclid_gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a


def extended_euclid(a, b):
    if b == 0:
        return a, 1, 0
    else:
        gcd, x1, y1 = extended_euclid(b, a % b)
        x = y1
        y = x1 - (a // b) * y1
        return gcd, x, y


def main_experiment1():
    gcd_result = euclid_gcd(1970, 1066)
    print(f"1. Euclid算法计算(1970, 1066)的最大公因子: {gcd_result}")
    gcd_val, x, y = extended_euclid(550, 1769)
    if gcd_val != 1:
        print("2. 550在模1769下的乘法逆元不存在")
    else:
        inverse = x % 1769
        print(f"2. 550在模1769下的乘法逆元: {inverse}")
        verification = (550 * inverse) % 1769
        print(f"  验证: 550 * {inverse} mod 1769 = {verification}")


if __name__ == "__main__":
    main_experiment1()
