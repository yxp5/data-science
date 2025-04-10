# class <Node>
# Author: yxp5

class Node():
    
    def __init__(self, entry):
        self.key = entry[0]
        self.value = entry[1]
        self.left = None
        self.right = None
        self.map = {}
        
    def info(self):
        left = ""
        if self.left == None:
            left = "None"
        else:
            left = self.left.info()
        
        right = "" 
        if self.right == None:
            right = "None"
        else:
            right = self.right.info()
            
        return f"({self.key}, {self.value}) with children {left} {right}"
    
    def get_mapping(self, node, string=""):
        if node.left == node.right: return self.map.update({node.key: string})
        
        if node.left != None: self.get_mapping(node.left, string+"0")
        if node.right != None: self.get_mapping(node.right, string+"1")
            
        return
    
    def mapping(self):
        if len(self.map) != 0: return self.map
        
        return self.get_mapping(self)
    
    def __str__(self):
        return self.info()

class NodeList:
    
    def __init__(self, frequency):
        self.list = list(map(lambda entry: Node(entry), frequency))
        self.length = len(self.list)

    def pop(self, index=0):
        if self.length == 0: return print("Cannot pop empty NodeList")
        self.length -= 1
        return self.list.pop(0)
    
    def insert(self, node):
        left = 0
        right = self.length - 1
        pointer = 0
        
        while left <= right:
            pointer = left + (right - left) // 2
            if self.list[pointer].value < node.value:
                left = pointer + 1
            elif self.list[pointer].value > node.value:
                right = pointer - 1
            else:
                break
            pointer = left + (right - left) // 2
        
        self.length += 1
        return self.list.insert(pointer + 1, node)
    
    def info(self):
        tmp = []
        for node in self.list:
            tmp.append(node.info())
        
        info = ", ".join(tmp)
        info = f"[{info}]"
        return info

    def __str__(self):
        return self.info()
























