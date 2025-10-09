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


def parser(data: str):
    eof = len(data)
    root = TreeNode(0, 0, eof)
    parent = root
    starts = []
    ends = []
    nodes = [root]
    for index in range(eof):
        char = data[index]
        if char == "{" or char == "[":
            starts.append(index)
        elif char == "}" or char == "]":
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
        nodes = sorted(nodes, key=lambda x: x.start)
        for i in range(1, len(nodes)):
            starta = nodes[i - 1].start
            startb = nodes[i].start
            enda = nodes[i - 1].end
            endb = nodes[i].end
            if starta < startb and enda > endb:
                garbage_node = TreeNode(-1, starta, startb, status=0)
                nodes.append(garbage_node)
        nodes = sorted(nodes, key=lambda x: x.end, reverse=True)
        for i in range(1, len(nodes)):
            starta = nodes[i - 1].start
            startb = nodes[i].start
            enda = nodes[i - 1].end
            endb = nodes[i].end
            if starta < startb and enda > endb:
                garbage_node = TreeNode(-1, endb, enda, status=0)
                nodes.append(garbage_node)
        nodes = sorted(nodes, key=lambda x: x.start)
        for i in range(1, len(nodes)):
            start = nodes[i - 1].end
            end = nodes[i].start
            if start < end:
                if start == 6 and end == 8:
                    print(nodes[i - 1].start, nodes[i - 1].end)
                    print(nodes[i].start, nodes[i].end)
                garbage_node = TreeNode(-1, start, end, status=0)
                nodes.append(garbage_node)

        return nodes

    nodes = create_garbage_nodes(nodes)

    def create_links(nodes):
        nodes = sorted(nodes, key=lambda x: x.start)
        for i in range(1, len(nodes)):
            node = nodes[i]
            for j in range(i - 1, -1, -1):
                parent = nodes[j]
                if parent.start <= node.start and parent.end >= node.end:
                    parent.add_child(node)
                    break

    create_links(nodes)
    nodes = sorted(nodes, key=lambda x: x.start)
    return nodes[0]
