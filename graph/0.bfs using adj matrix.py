from collections import deque

def bfs(graph, start):
    visited = set()  # To keep track of visited nodes
    queue = deque([start])  # Initialize the queue with the starting node
    order = []  # List to store the order of traversal

    while queue:
        node = queue.popleft()  # Dequeue a node
        if node not in visited:
            visited.add(node)  # Mark the node as visited
            order.append(node)  # Append it to the traversal order

            # Enqueue all unvisited adjacent nodes
            for neighbor, connected in enumerate(graph[node]):
                if connected and neighbor not in visited:
                    queue.append(neighbor)

    return order

# Example usage:
graph = [
    [0, 1, 1, 0, 0, 0],  # Node A
    [1, 0, 0, 1, 1, 0],  # Node B
    [1, 0, 0, 0, 0, 1],  # Node C
    [0, 1, 0, 0, 0, 0],  # Node D
    [0, 1, 0, 0, 0, 1],  # Node E
    [0, 0, 1, 0, 1, 0]   # Node F
]

start_node = 0  # Starting from node A (index 0)
order = bfs(graph, start_node)
print("BFS Traversal Order:", order)
