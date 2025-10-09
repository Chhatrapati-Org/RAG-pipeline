class TreeNode:
    def __init__(self, id, start, end, status=1):
        self.data = id
        self.start = start
        self.end = end
        self.status = status
        self.children = []
        self.parent = None

    def add_child(self, child_node):
        child_node.parent = self
        self.children.append(child_node)

    def remove_child(self, child_node):
        child_node.parent = None
        self.children.remove(child_node)

    def get_level(self):
        level = 0
        p = self.parent
        while p:
            level += 1
            p = p.parent
        return level

    def print_tree(self):
        prefix = " " * self.get_level() * 3 + "|-- " if self.parent else ""
        print(prefix + str(self.start) + "\t" + str(self.end))
        if self.children:
            for child in self.children:
                child.print_tree()


def parser(data:str):
    # Make sure data is a string
    if not isinstance(data, str):
        print(f"Warning: parser received non-string input of type {type(data)}")
        data = str(data)
        
    eof = len(data)-1
    root = TreeNode(0,0,eof)
    parent = root
    starts = []
    ends = []
    nodes = [root]
    for index in range(eof+1):
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
                    nodes.append(child)
                    break
                start = starts.pop(j - 1)
                child = TreeNode(len(parent.children), start, end)
                nodes.append(child)
                break
            if j == len(starts) - 1:
                start = starts.pop()
                child = TreeNode(len(parent.children), start, end)
                nodes.append(child)
                break
    while starts:
        start = starts.pop()
        child = TreeNode(len(parent.children), start, eof)
        nodes.append(child)

    def create_garbage_nodes(nodes):
        # Safety check - if no nodes or just one node, return as is
        if len(nodes) <= 1:
            return nodes
            
        # Ensure all nodes have integer values for start and end
        for node in nodes:
            try:
                node.start = int(node.start)
                node.end = int(node.end)
            except (TypeError, ValueError) as e:
                print(f"Warning: Non-integer node indices found: start={node.start}, end={node.end}")
                # Set to safe defaults
                node.start = 0
                node.end = 0
                node.status = 0  # Mark as invalid
        
        nodes = sorted(nodes, key=lambda x: x.start)
        for i in range(1, len(nodes)):
            starta = int(nodes[i - 1].start)
            startb = int(nodes[i].start)
            enda = int(nodes[i - 1].end)
            endb = int(nodes[i].end)
            if starta < startb and enda > endb:
                garbage_node = TreeNode(-1, starta, startb, status=0)
                nodes.append(garbage_node)
                
        nodes = sorted(nodes, key=lambda x: x.end, reverse=True)
        for i in range(1, len(nodes)):
            starta = int(nodes[i - 1].start)
            startb = int(nodes[i].start)
            enda = int(nodes[i - 1].end)
            endb = int(nodes[i].end)
            if starta < startb and enda > endb:
                garbage_node = TreeNode(-1, endb, enda, status=0)
                nodes.append(garbage_node)
                
        nodes = sorted(nodes, key=lambda x: x.start)
        for i in range(1, len(nodes)):
            start = int(nodes[i - 1].end)
            end = int(nodes[i].start)
            if start < end:
                garbage_node = TreeNode(-1, start, end, status=0)
                nodes.append(garbage_node)

        return nodes

    nodes = create_garbage_nodes(nodes)

    def create_links(nodes):
        # Safety check - if no nodes or just one node, return
        if len(nodes) <= 1:
            return
            
        try:
            nodes = sorted(nodes, key=lambda x: x.start)
            for i in range(1, len(nodes)):
                node = nodes[i]
                for j in range(i - 1, -1, -1):
                    parent = nodes[j]
                    # Make sure we're comparing integers
                    node_start = int(node.start)
                    node_end = int(node.end)
                    parent_start = int(parent.start)
                    parent_end = int(parent.end)
                    
                    if parent_start <= node_start and parent_end >= node_end:
                        parent.add_child(node)
                        break
        except Exception as e:
            print(f"Error in create_links: {e}")
            import traceback
            traceback.print_exc()

    create_links(nodes)
    nodes = sorted(nodes, key=lambda x: x.start)
    return nodes[0]
