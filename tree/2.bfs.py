
from collections import deque

class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def bfs(root):
    if not root:
        return
    
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        
        # Process the current node (print it, for example)
        print(node.value)
        
        # Enqueue left child if it exists
        if node.left:
            queue.append(node.left)
        
        # Enqueue right child if it exists
        if node.right:
            queue.append(node.right)

# Example usage:
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)
root.right.left = TreeNode(6)
root.right.right = TreeNode(7)

bfs(root)
