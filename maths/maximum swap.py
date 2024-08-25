class Solution:
    def maximumSwap(self, num: int) -> int:
        s=str(num)
        first= int(s[0])
        best= 'No swap'
        for i in range(1,len(s)):
            if int(s[i]) >first:
                first = int(s[i])
        if first != int(s[0]):
            best =first
            
        for i in range(len(s)):
            if s[i] == 'No swap':
                break
            if s[i] == best:
                s[0],s[i] = s[i],s[0]
        return int(s)

s=Solution()
num = 2736
s.maximumSwap(num)
        