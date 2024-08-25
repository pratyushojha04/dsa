def binary_to_decimal(binary_string):
    decimal = 0
    for i in range(len(binary_string)):
        decimal += int(binary_string[i]) * (2 ** (len(binary_string) - 1 - i))
    return decimal

binary_string = "101010"
decimal_number = binary_to_decimal(binary_string)
print(decimal_number)  # Output: 42

# method2
binary_string = "101010"
decimal_number = int(binary_string, 2)
print(decimal_number)  # Output: 42
