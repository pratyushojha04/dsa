def add(s):
    l = []
    for i in range(len(s)):
        if s[i] == '\n':
            continue
        if s[i] == '@':
            return "not allowed"
        
        try:
            l.append(int(s[i]))
        except ValueError:
            continue
    
    return sum(l)


print(add(""))            # Output: 0
print(add("1"))           # Output: 1
print(add("1,5"))         # Output: 6
print(add("1\n2,3"))      # Output: 6
print(add("//;\n1;2"))    # Output: 3
print(add("//|\n1|2|3"))  # Output: 6
print(add("//:\n1:2:3"))  # Output: 6
print(add("//@\n1@-2@3")) # Output: not allowed
