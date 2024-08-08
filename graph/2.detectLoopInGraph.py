from typing import List

class Solution:
    def __init__(self):
        self.vis = []

    def dfs(self, node: int, parent: int, adj: List[List[int]]) -> bool:
        self.vis[node] = 1
        # Visit adjacent nodes
        for adjacentNode in adj[node]:
            # Unvisited adjacent node
            if not self.vis[adjacentNode]:
                if self.dfs(adjacentNode, node, adj):
                    return True
            # Visited node but not a parent node
            elif adjacentNode != parent:
                return True
        return False

    # Function to detect cycle in an undirected graph.
    def isCycle(self, V: int, adj: List[List[int]]) -> bool:
        self.vis = [0] * V
        # For graph with connected components
        for i in range(V):
            if not self.vis[i]:
                if self.dfs(i, -1, adj):
                    return True
        return False

# Example usage:
if __name__ == "__main__":
    # V = 4, E = 2
    adj = [[1,1,0],[1,1,0],[0,0,1]]
    solution = Solution()
    ans = solution.isCycle(4, adj)
    if ans:
        print("1")
    else:
        print("0")
