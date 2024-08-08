class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def dfs_recursive(node):
    if not node:
        return
    
    # Process the current node (print it, for example)
    
    
    # Recursively visit the left child
    dfs_recursive(node.left)
    print(node.value)
    # Recursively visit the right child
    dfs_recursive(node.right)

# Example usage:
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)
root.right.left = TreeNode(6)
root.right.right = TreeNode(7)

dfs_recursive(root)
