# Based on Appendix E, version 2.1 8-7-2024
# No considerations have been given to exclusive conditions yet

defining_set = (23, 16, 4, 7, 6, 18, 14, 2, 31, 21, 5, 8, 28)

non_defining_set = (12, 11, 19, 30, 15, 34, 33, 13, 17, 29, 20, 1, 25, 24, 32, 35, 36, 10, 3, 27, 9, 22, 26)

if __name__ == "__main__":
    print(len(defining_set) + len(non_defining_set))
    print(sum(defining_set) + sum(non_defining_set) == sum(list(range(1, 37))))
