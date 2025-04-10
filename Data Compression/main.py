# Data Compression
# Author: yxp5

import math
from binarytree import *

def TextToHex(content):
    hex_string = ""
    
    for char in content:
        binary = bin(ord(char))[2:]
        binary = "0" * (8 - len(binary)) + binary
        hex1 = hex(int(binary[:4], 2))[2:]
        hex2 = hex(int(binary[4:], 2))[2:]
        hex_string += hex1
        hex_string += hex2
    
    return hex_string

def BinToHex(compressed):
    length = len(compressed)
    hex_string = ""
    
    for i in range(4, length + 4, 4):
        hex_string += hex(int(compressed[i-4:i], 2))[2:]
    
    return hex_string

def Frequency(hex_string):
    frequency = {}
    for h in hex_string:
        frequency.update({h: frequency.setdefault(h, 0) + 1})
    
    string_length = len(hex_string)
    for key in frequency.keys():
        frequency[key] = round(frequency[key] / string_length * 100, 4)
    
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

def CompressionMapping(frequency):
    root = EncodingTree(frequency)
    root.mapping()
    return root.map

def Compress(string, mapping):
    compressed = ""
    
    for char in string:
        compressed += mapping[char]

    return compressed

def Compare(original, compressed):
    # Each character is 1 byte = 8 bits
    # Each float is 4 bytes = 32 bits (the usual size of a single parameter in LLM)
    # Each double is 8 bytes = 64 bits
    # Each long double is 16 bytes = 128 bits
    o_size = len(original) * 32
    c_size = len(compressed)
    reduction = round((c_size-o_size)/o_size*100, 2)
    
    print(f"The original size is {o_size} bits, the compressed size is {c_size} bits, {reduction}% reduction!")
    return reduction

def Recover(filename, mapping):
    inv_mapping = {v: k for k, v in mapping.items()}
    
    fp = open(f"{filename}_compressed.txt", "r")
    compressed = fp.read()
    fp.close()
    
    fp = open(f"{filename}_recover.txt", "w")
    recover = ""
    
    key = ""
    hexs = []
    for bit in compressed:
        key += bit
        if key in inv_mapping:
            hexs.append(inv_mapping[key])
            eight = len(hexs) == 8
            if eight:
                binary = ""
                for h in hexs :
                    tmp = bin(int(h, 16))[2:]
                    pad = "0" * (4 - len(tmp))
                    binary = binary + pad + tmp
                c = chr(int(binary, 2))
                recover += c
                hexs = []
            key = ""
    
    fp.write(recover)
    return recover

filename = "test.txt"
original = ""
compressed = ""
recover = ""
frequency = None
mappings = []
paddings = []
repetition = 1

fp = open(f"{filename}", "r")
original = fp.read()
content = original
fp.close()
count = 0

hex_string = TextToHex(content)
while count < repetition:
    frequency = Frequency(hex_string)
    mapping = CompressionMapping(frequency)
    compressed = Compress(hex_string, mapping)
    
    padding = "0" * (4 - len(compressed) % 4)
    compressed = padding + compressed
    
    hex_string = BinToHex(compressed)
    
    paddings.append(padding)
    mappings.append(mapping)
    count += 1

reduction = Compare(original, compressed)
efficiency = (100 + reduction) / 100

power = 1 / efficiency / efficiency
print(f"Power increase of {round(power, 2)}x")























