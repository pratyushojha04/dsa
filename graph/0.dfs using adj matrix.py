def dfs_matrix(graph, start):
    visited = set()  # To keep track of visited nodes
    stack = [start]  # Initialize the stack with the starting node
    order = []  # List to store the order of traversal

    while stack:
        node = stack.pop()  # Pop a node from the stack
        if node not in visited:
            visited.add(node)  # Mark the node as visited
            order.append(node)  # Append it to the traversal order

            # Push all unvisited adjacent nodes onto the stack
            for neighbor in range(len(graph[node]) - 1, -1, -1):
                if graph[node][neighbor] and neighbor not in visited:
                    stack.append(neighbor)

    return order

# Example usage:
graph_matrix = [
    [0, 1, 1, 0, 0, 0],  # Node A
    [1, 0, 0, 1, 1, 0],  # Node B
    [1, 0, 0, 0, 0, 1],  # Node C
    [0, 1, 0, 0, 0, 0],  # Node D
    [0, 1, 0, 0, 0, 1],  # Node E
    [0, 0, 1, 0, 1, 0]   # Node F
]

start_node_matrix = 0  # Starting from node A (index 0)
order_matrix = dfs_matrix(graph_matrix, start_node_matrix)
print("DFS Traversal Order (Adjacency Matrix):", order_matrix)
