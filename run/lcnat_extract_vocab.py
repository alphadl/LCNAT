#!/usr/bin/env python3
"""Extract one token per line from fairseq dict (format: 'token count')."""
import sys

def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: lcnat_extract_vocab.py <dict.txt> <output.vocab>")
    with open(sys.argv[1], "r", encoding="utf-8") as fin:
        with open(sys.argv[2], "w", encoding="utf-8") as fout:
            for line in fin:
                idx = line.rfind(" ")
                word = (line[:idx] if idx >= 0 else line).strip()
                fout.write(word + "\n")

if __name__ == "__main__":
    main()
