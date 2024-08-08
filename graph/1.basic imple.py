from collections import deque

class Graph:
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.adj_list = [[] for _ in range(num_vertices)]
    
    def add_edge(self, src, dest):
        self.adj_list[src].append(dest)
        self.adj_list[dest].append(src)  # Uncomment for undirected graph
    
    def bfs(self, start):
        visited = [False] * self.num_vertices
        queue = deque([start])
        visited[start] = True
        
        while queue:
            node = queue.popleft()
            print(node, end=" ")
            
            for neighbor in self.adj_list[node]:
                if not visited[neighbor]:
                    queue.append(neighbor)
                    visited[neighbor] = True

# Example usage
graph = Graph(5)
graph.add_edge(0, 1)
graph.add_edge(0, 4)
graph.add_edge(1, 2)
graph.add_edge(1, 3)
graph.add_edge(1, 4)
graph.add_edge(2, 3)
graph.add_edge(3, 4)

graph.bfs(0)
