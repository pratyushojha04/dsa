def dfs(graph, start):
    visited = set()  # To keep track of visited nodes
    stack = [start]  # Initialize the stack with the starting node
    order = []  # List to store the order of traversal

    while stack:
        node = stack.pop()  # Pop a node from the stack
        if node not in visited:
            visited.add(node)  # Mark the node as visited
            order.append(node)  # Append it to the traversal order

            # Push all unvisited adjacent nodes onto the stack
            for neighbor in reversed(graph[node]):
                if neighbor not in visited:
                    stack.append(neighbor)

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
print("DFS Traversal Order (Adjacency List):", dfs(graph, start_node))
