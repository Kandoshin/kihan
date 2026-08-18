#问题1，2
# table_len = 8
# key = "567"
# key_seq_len = 2
# s_table = list(range(table_len))
# k_table = []
# for i in range(table_len):
#     k_table.append(int(key[i % len(key)]))
# j = 0
# for i in range(table_len):
#     j = (j + s_table[i] + k_table[i]) % table_len
#     s_table[i], s_table[j] = s_table[j], s_table[i]
#
#     if i == 7:
#         last_swap = (i, j)
#         final_s_table = s_table.copy()
#
# i = j = 0
# key_seq = []
# for _ in range(key_seq_len):
#     i = (i + 1) % table_len
#     j = (j + s_table[i]) % table_len
#     s_table[i], s_table[j] = s_table[j], s_table[i]
#     key_byte = s_table[(s_table[i] + s_table[j]) % table_len]
#     key_seq.append(key_byte)
#
# print("问题1:")
# print(f"最后一次循环(i={last_swap[0]})，将S({last_swap[0]})与S({last_swap[1]})互换")
# print(f"随机化后的S表: {final_s_table}\n")
#
# print("问题2:")
# print(f"第一个密钥字: {key_seq[0]} (二进制: {bin(key_seq[0])[2:].zfill(3)})")
# print(f"第二个密钥字: {key_seq[1]} (二进制: {bin(key_seq[1])[2:].zfill(3)})")

#问题3
def rc4(data, key):
    n = 256
    s = list(range(n))
    k = [ord(key[i % len(key)]) for i in range(n)]
    j = 0
    for i in range(n):
        j = (j + s[i] + k[i]) % n
        s[i], s[j] = s[j], s[i]

    i = j = 0
    result = []
    for char in data:
        i = (i + 1) % n
        j = (j + s[i]) % n
        s[i], s[j] = s[j], s[i]
        k = s[(s[i] + s[j]) % n]
        result.append(chr(ord(char) ^ k))

    return ''.join(result)


def bytes_to_hex(data):
    return ''.join(f"{ord(c):02x}" for c in data)


def hex_to_bytes(hex_str):
    return ''.join(chr(int(hex_str[i:i + 2], 16)) for i in range(0, len(hex_str), 2))

key = "30216"
plaintext = "xiaojinlei"

ciphertext = rc4(plaintext, key)
hex_cipher = bytes_to_hex(ciphertext)

decrypted = rc4(ciphertext, key)

print("\n问题3:")
print(f"明文: {plaintext}")
print(f"密钥: {key}")
print(f"密文(16进制): {hex_cipher}")
print(f"解密结果: {decrypted}")

assert decrypted == plaintext, "解密结果与原始明文不一致"
print("验证成功：解密结果与原始明文一致")