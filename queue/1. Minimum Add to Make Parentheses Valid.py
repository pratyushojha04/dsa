from collections import deque

class Solution:
    def minAddToMakeValid(self, s: str) -> int:
        q=deque()
        count=0
        for i in range(len(s)):
            if s[i] == '(':
                q.append(s[i])
            if s[i] == ')' and len(q) == 0:
                count+=1
            if s[i] == ')' and len(q) != 0:
                q.pop()
            
        return len(q)
        

c=Solution()
s = "())"
c.minAddToMakeValid(s)