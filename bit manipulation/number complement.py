
class Solution:
    def findComplement(self, num: int) -> int:
        s=bin(num)[2:]
        s1=''
        for i in s:
            if i == '0':
                s1+='1'
            else:
                s1+='0'
        return int(s1,2)
    
s=Solution()
s.findComplement(5)