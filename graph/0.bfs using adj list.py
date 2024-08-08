from collections import deque

def bfs(graph, start):
    visited = set()       # To keep track of visited nodes
    queue = deque([start])  # Initialize the queue with the starting node
    order = []           # List to store the order of traversal

    while queue:
        node = queue.popleft()  # Dequeue a node
        if node not in visited:
            visited.add(node)   # Mark the node as visited
            order.append(node)  # Append it to the traversal order
            
            # Enqueue all unvisited adjacent nodes
            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
    
    return order

# Example usage:
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

start_node = 'A'
print("BFS Traversal Order:", bfs(graph, start_node))
