# /// script
# dependencies = [
#   "numpy",
# ]
# ///

import numpy as np
import subprocess
import sys
import os
import argparse

def load_vec(line):
    return np.fromstring(line, sep=' ')

def dump(path, opt):
    proc = subprocess.Popen(['fasttext', 'dump', path, opt], 
                             stdout=subprocess.PIPE, 
                             stderr=subprocess.PIPE,
                             text=True)
    return proc

def main():
    parser = argparse.ArgumentParser(description='extract telling tokens for a fasttext label')
    parser.add_argument('model', help='path to the fasttext model (.bin)')
    parser.add_argument('label', help='the label to analyze (e.g., __label__1)')
    parser.add_argument('-n', '--top_n', type=int, default=100, help='number of tokens to extract')
    
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"error: model file {args.model} not found.")
        sys.exit(1)

    print(f"reading dictionary from {args.model}...")
    words = []
    labels = []
    with dump(args.model, 'dict').stdout as f:
        head = f.readline()
        if not head:
            print("error: could not read dictionary dump.")
            sys.exit(1)
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            if parts[-1] == 'word':
                words.append(" ".join(parts[:-2]))
            elif parts[-1] == 'label':
                labels.append(" ".join(parts[:-2]))

    if args.label not in labels:
        print(f"error: label '{args.label}' not found in model.")
        print(f"available labels: {', '.join(labels)}")
        sys.exit(1)
    
    idx = labels.index(args.label)
    print(f"found label '{args.label}' at index {idx}.")

    print(f"reading output vectors from {args.model}...")
    with dump(args.model, 'output').stdout as f:
        head = f.readline()
        for i in range(idx):
            f.readline()
        ln = f.readline()
        if not ln:
            print(f"error: could not read vector for label {args.label}.")
            sys.exit(1)
        l_vec = load_vec(ln)

    print(f"processing input vectors and computing dot products...")
    top = []
    proc = dump(args.model, 'input')
    try:
        f = proc.stdout
        head = f.readline()
        if not head:
            print("error: could not read input vectors dump.")
            sys.exit(1)
        
        n_vecs, dim = map(int, head.split())
        
        for i in range(n_vecs):
            line = f.readline()
            if not line: break
            vec = load_vec(line)
            score = np.dot(vec, l_vec)
            if i < len(words):
                top.append((words[i], score))
            
            if i % 100000 == 0 and i > 0:
                print(f"  processed {i}/{n_vecs} vectors...", end='\r', flush=True)
    finally:
        proc.terminate()

    print("\nsorting results...")
    top.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\ntop {args.top_n} tokens for {args.label}:")
    for word, score in top[:args.top_n]:
        print(f"{word}\t{score:.4f}")

if __name__ == "__main__":
    main()
