# Data Compression
# Author: yxp5

import math
from binarytree import *

def Frequency(string):
    frequency = {}
    for char in string:
        frequency.update({char: frequency.setdefault(char, 0) + 1})
    
    string_length = len(string)
    for key in frequency.keys():
        frequency[key] = round(frequency[key] / string_length * 100, 2)
    
    frequency = list(frequency.items())
    frequency.sort(key=lambda tup: tup[1])
    return frequency

def EncodingTree(frequency):
    nodelist = NodeList(frequency)
    
    while nodelist.length > 1:
        node0 = nodelist.pop()
        node1 = nodelist.pop()
        node01 = Node((f"{node0.key} {node1.key}", round(node0.value + node1.value, 2)))
        node01.left = node0
        node01.right = node1
        
        nodelist.insert(node01)
    
    root = nodelist.pop()
    return root

def CompressionMapping(root):
    root.mapping()
    return root.map

def Compress(string, mapping, filename):
    fp = open(f"{filename}_compressed.txt", "w")
    compressed = ""
    
    for char in string:
        compressed += mapping[char]
    
    fp.write(compressed)
    fp.close()
    return compressed

def Compare(original, compressed):
    # Each character is 1 byte = 8 bits
    # Each float is 4 bytes = 32 bits (the usual size of a single parameter in LLM)
    # Each double is 8 bytes = 64 bits
    # Each long double is 16 bytes = 128 bits
    o_size = len(original) * 8 * 4
    c_size = len(compressed)
    reduction = round((c_size-o_size)/o_size*100, 2)
    
    print(f"The original size is {o_size} bits, the compressed size is {c_size} bits, {reduction}% reduction!")
    return reduction

def Recover(filename, mapping):
    fp = open(f"{filename}_compressed.txt", "r")
    compressed = fp.read()
    fp.close()
    
    fp = open(f"{filename}_recover.txt", "w")
    recover = ""
    
    key = ""
    for bit in compressed:
        key += bit
        if key in mapping:
            recover += mapping[key]
            key = ""
            continue
    
    fp.write(recover)
    return recover

def Main():
    global mapping, original, compressed, frequency, mapping, recover
    filename = input(f"Choose the file you wish to compress: ")
    fp = open(f"{filename}.txt", "r")
    content = fp.read()
    original = content
    fp.close()
    
    frequency = Frequency(content)
    root = EncodingTree(frequency)
    mapping = CompressionMapping(root)
    compressed = Compress(content, mapping, filename)
    
    reduction = Compare(original, compressed)
    efficiency = (100 + reduction) / 100
    power = 1 / efficiency / efficiency
    
    print(f"The increase in power is {round(power, 2)}x")
    
    inv_mapping = {v: k for k, v in mapping.items()}
    recover = Recover(filename, inv_mapping)
    return

if __name__ == "__main__":
    original = ""
    compressed = ""
    recover = ""
    frequency = None
    mapping = {}
    Main()





























