class Solution:
    def isin(self, s1: str, s2: str) -> str:
        d = {}
        for char in s1:
            d[char] = d.get(char, 0) + 1
        
        for char in s2:
            a = d.get(char, 0) or 1
            if a > s2.count(char):
                return False
        
        return True

    def minWindow(self, s: str, t: str) -> str:
        if len(s) < len(t): return ""
        if not self.isin(t, s): return ""

        left, right = 0, len(s)

        while self.isin(t, s[left:right]):
            if self.isin(t, s[left:right-1]):
                right = right - 1
            elif self.isin(t, s[left+1:right]):
                left = left + 1
            else:
                return s[left:right]

a = Solution()
b = "ABC"
c = "ADOBECODEBANC"

x = a.isin(b, c)
y = a.minWindow(c, b)