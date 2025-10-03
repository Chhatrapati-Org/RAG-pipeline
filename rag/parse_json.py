class TreeNode:
    """
    A class representing a node in a tree data structure.
    """
    def __init__(self, id, start, end, status = 1):
        """
        Initializes a new node.
        
        Args:
            status: 0 if incomplete file 1 if complete dict found
        """
        self.data = id
        self.start = start
        self.end = end
        self.status = status
        self.children = []
        self.parent = None

    def add_child(self, child_node):
        """
        Adds a child node to the current node.
        
        Args:
            child_node (TreeNode): The node to be added as a child.
        """
        child_node.parent = self  # Set the current node as the parent of the child
        self.children.append(child_node)

    def remove_child(self, child_node):
        """
        Removes a child node from the current node.
        
        Args:
            child_node (TreeNode): The node to be removed from the children.
        """
        child_node.parent = None  # Remove the parent reference from the child
        self.children.remove(child_node)

    def get_level(self):
        """
        Returns the level (or depth) of the current node in the tree.
        The root node is at level 0.
        """
        level = 0
        p = self.parent
        while p:
            level += 1
            p = p.parent
        return level

    def print_tree(self):
        """
        Prints the tree structure starting from the current node.
        """
        prefix = ' ' * self.get_level() * 3 + '|-- ' if self.parent else ""
        print(prefix + self.data)
        if self.children:
            for child in self.children:
                child.print_tree()


def parser(data:str):
    eof = len(data)
    root = TreeNode(0,0,eof)
    parent = root
    starts = []
    ends = []
    for index in range(eof):
        char = data[index]
        if char == '{' or char == '[':
            starts.append(index)
        elif char == '}' or char == ']':
            ends.append(index)

    while ends:
        end = ends.pop(0)
        for j in range(len(starts)):
            index = starts[j]
            if index > end:
                if j == 0:
                    start = 0
                    child = TreeNode(len(parent.children), start, end)
                    parent.add_child(child)
                    break
                start = starts.pop(j - 1)
                child = TreeNode(len(parent.children), start, end)
                parent.add_child(child)
                break
            if j == len(starts) - 1:
                start = starts.pop()
                child = TreeNode(len(parent.children), start, end)
                parent.add_child(child)
                parent = parent.parent
                break
    while starts:
        start = starts.pop()
        child = TreeNode(len(parent.children), start, eof)
        parent.add_child(child)
        parent = child

    return root
