import re

class StringCalculator:
    def add(self, numbers: str) -> int:
        if not numbers:
            return 0
        if numbers.startswith("//"):
            delimiter, numbers = self._parse_custom_delimiter(numbers)
        else:
            delimiter = ',|\n'  
        numbers_list = re.split(delimiter, numbers)
        total = 0
        negatives = []
        for num in numbers_list:
            if num:
                number = int(num)
                if number < 0:
                    negatives.append(number)
                else:
                    total += number
        if negatives:
            raise ValueError(f"Negative numbers not allowed: {', '.join(map(str, negatives))}")

        return total

    def _parse_custom_delimiter(self, numbers: str):
        first_newline_index = numbers.index('\n')
        delimiter_part = numbers[2:first_newline_index]
        numbers_part = numbers[first_newline_index + 1:]

        # Handle multiple-character delimiters
        if delimiter_part.startswith('[') and delimiter_part.endswith(']'):
            delimiter = re.escape(delimiter_part[1:-1])  # Escape special characters
        else:
            delimiter = re.escape(delimiter_part)

        return delimiter, numbers_part

# Example usage
calc = StringCalculator()

print(calc.add(""))             # Output: 0
print(calc.add("1"))            # Output: 1
print(calc.add("1,5"))          # Output: 6
print(calc.add("1\n2,3"))       # Output: 6
print(calc.add("//;\n1;2"))     # Output: 3
print(calc.add("//[***]\n1***2***3"))  # Output: 6

try:
    print(calc.add("//;\n1;-2;3"))  # Raises an exception
except ValueError as e:
    print(e)  # Output: Negative numbers not allowed: -2
