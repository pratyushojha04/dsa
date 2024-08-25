def decimal_to_binary(n):
    binary = ''
    while n > 0:
        binary = str(n % 2) + binary
        n = n // 2
    return binary

decimal_number = 42
binary_representation = decimal_to_binary(decimal_number)
print(binary_representation)  # '101010'


#method2
decimal_number = 42
binary_representation = bin(decimal_number)  # '0b101010'
binary_string = bin(decimal_number)[2:]  # '101010' (without the '0b' prefix)
print(binary_string)

