# Add the necessary imports here
import pandas as pd

import numpy as np
import math

import torch

# from PIL import Image
# import torchvision.transforms as transforms
from utils import *

# import cv2
import matplotlib.pyplot as plt

from collections import Counter
import heapq


def solve_cv_easy(test_case: tuple) -> list:
    """
    This function takes a tuple as input and returns a list as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A numpy array representing a shredded image.
        - An integer representing the shred width in pixels.

    Returns:
    list: A list of integers representing the order of shreds. When combined in this order, it builds the whole image.
    """
    img, shred_width = input
    img = np.array(img)

    height, width = img.shape[:2]

    out = [0]

    master = img[:, 0:shred_width, :]
    shreds = []
    shreds_ind = []

    for i in range(int(width / shred_width - 1)):
        i = i + 1
        shreds.append(img[:, i * shred_width : (i + 1) * shred_width, :])
        shreds_ind.append(i)

    while len(shreds) > 0:
        best_value = 999999999999
        best_ind = 0

        for i, shred in enumerate(shreds):
            score = np.sum(master[:, -1, :] - shreds[i][:, 0, :])
            if score < best_value:
                best_value = score
                best_ind = i

        out.append(shreds_ind[best_ind])

        master = shreds[best_ind]

        shreds.pop(best_ind)
        shreds_ind.pop(best_ind)

    return out


def solve_cv_medium(input: tuple) -> list:
    # combined_image_array, patch_image_array = test_case
    # combined_image = np.array(combined_image_array, dtype=np.uint8)
    # patch_image = np.array(patch_image_array, dtype=np.uint8)
    """
    This function takes a tuple as input and returns a list as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A numpy array representing the RGB base image.
        - A numpy array representing the RGB patch image.

    Returns:
    list: A list representing the real image.
    """
    return []


def solve_cv_hard(input: tuple) -> int:
    # extracted_question, image = test_case
    # image = np.array(image)
    """
    This function takes a tuple as input and returns an integer as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A string representing a question about an image.
        - An RGB image object loaded using the Pillow library.

    Returns:
    int: An integer representing the answer to the question about the image.
    """
    return 2


def solve_ml_easy(input: pd.DataFrame) -> list:
    data = pd.DataFrame(data)

    """
    This function takes a pandas DataFrame as input and returns a list as output.

    Parameters:
    input (pd.DataFrame): A pandas DataFrame representing the input data.

    Returns:
    list: A list of floats representing the output of the function.
    """
    return []


def solve_ml_medium(input: list) -> int:
    """
    This function takes a list as input and returns an integer as output.

    Parameters:
    input (list): A list of signed floats representing the input data.

    Returns:
    int: An integer representing the output of the function.
    """

    def distance(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def is_between(a, c, b):
        return distance(a, c) + distance(c, b) == distance(a, b)

    pnt = np.array(input)

    df = pd.read_csv("Solvers/MlMediumTrainingData.csv")
    x = df.drop("class", axis=1)
    X = x.values

    for i in range(2029):
        if is_between(X[i], pnt, X[i + 1]):
            return 0

    return -1


def solve_sec_medium(input) -> str:
    """
    This function takes a torch.Tensor as input and returns a string as output.

    Parameters:
    input (torch.Tensor): A torch.Tensor representing the image that has the encoded message.

    Returns:
    str: A string representing the decoded message from the image.
    """
    try:
        # data preprocessing
        input = np.array(input, dtype=np.float32)
        input = np.transpose(input, (2, 0, 1))
        input = torch.tensor(input)

        # logic
        batched_image_tensor = torch.stack(list(input))
        batched_image_tensor = batched_image_tensor.unsqueeze(0)
        decoded_message = decode(batched_image_tensor)
        return decoded_message
    except Exception as e:
        print(str(e))
        return ""


def solve_sec_hard(input: tuple) -> str:
    pc1 = [
        56,
        48,
        40,
        32,
        24,
        16,
        8,
        0,
        57,
        49,
        41,
        33,
        25,
        17,
        9,
        1,
        58,
        50,
        42,
        34,
        26,
        18,
        10,
        2,
        59,
        51,
        43,
        35,
        62,
        54,
        46,
        38,
        30,
        22,
        14,
        6,
        61,
        53,
        45,
        37,
        29,
        21,
        13,
        5,
        60,
        52,
        44,
        36,
        28,
        20,
        12,
        4,
        27,
        19,
        11,
        3,
    ]
    pc2 = [
        13,
        16,
        10,
        23,
        0,
        4,
        2,
        27,
        14,
        5,
        20,
        9,
        22,
        18,
        11,
        3,
        25,
        7,
        15,
        6,
        26,
        19,
        12,
        1,
        40,
        51,
        30,
        36,
        46,
        54,
        29,
        39,
        50,
        44,
        32,
        47,
        43,
        48,
        38,
        55,
        33,
        52,
        45,
        41,
        49,
        35,
        28,
        31,
    ]
    ip = [
        57,
        49,
        41,
        33,
        25,
        17,
        9,
        1,
        59,
        51,
        43,
        35,
        27,
        19,
        11,
        3,
        61,
        53,
        45,
        37,
        29,
        21,
        13,
        5,
        63,
        55,
        47,
        39,
        31,
        23,
        15,
        7,
        56,
        48,
        40,
        32,
        24,
        16,
        8,
        0,
        58,
        50,
        42,
        34,
        26,
        18,
        10,
        2,
        60,
        52,
        44,
        36,
        28,
        20,
        12,
        4,
        62,
        54,
        46,
        38,
        30,
        22,
        14,
        6,
    ]
    expansion_table = [
        31,
        0,
        1,
        2,
        3,
        4,
        3,
        4,
        5,
        6,
        7,
        8,
        7,
        8,
        9,
        10,
        11,
        12,
        11,
        12,
        13,
        14,
        15,
        16,
        15,
        16,
        17,
        18,
        19,
        20,
        19,
        20,
        21,
        22,
        23,
        24,
        23,
        24,
        25,
        26,
        27,
        28,
        27,
        28,
        29,
        30,
        31,
        0,
    ]
    S1 = [
        [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
        [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
        [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
        [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13],
    ]
    S2 = [
        [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
        [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
        [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
        [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9],
    ]
    S3 = [
        [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
        [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
        [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
        [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12],
    ]
    S4 = [
        [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
        [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
        [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
        [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14],
    ]
    S5 = [
        [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
        [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
        [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
        [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3],
    ]
    S6 = [
        [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
        [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
        [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
        [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13],
    ]
    S7 = [
        [4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
        [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
        [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
        [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12],
    ]
    S8 = [
        [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
        [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
        [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
        [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11],
    ]
    p_sbox = [
        15,
        6,
        19,
        20,
        28,
        11,
        27,
        16,
        0,
        14,
        22,
        25,
        4,
        17,
        30,
        9,
        1,
        7,
        23,
        13,
        31,
        26,
        2,
        8,
        18,
        12,
        29,
        5,
        21,
        10,
        3,
        24,
    ]
    fp = [
        39,
        7,
        47,
        15,
        55,
        23,
        63,
        31,
        38,
        6,
        46,
        14,
        54,
        22,
        62,
        30,
        37,
        5,
        45,
        13,
        53,
        21,
        61,
        29,
        36,
        4,
        44,
        12,
        52,
        20,
        60,
        28,
        35,
        3,
        43,
        11,
        51,
        19,
        59,
        27,
        34,
        2,
        42,
        10,
        50,
        18,
        58,
        26,
        33,
        1,
        41,
        9,
        49,
        17,
        57,
        25,
        32,
        0,
        40,
        8,
        48,
        16,
        56,
        24,
    ]
    sboxs = [S1, S2, S3, S4, S5, S6, S7, S8]

    final_plain_txt = ""
    res = ""
    res2 = ""

    def toBinary(a):
        return bin(int(a, 16))[2:].zfill(64)

    def sbox(stable, i, ans):
        return stable[int(ans[i][0] + ans[i][5], 2)][int(ans[i][1:5], 2)]

    def permutation(size, old, pt):
        z = ""
        for i in range(0, size):
            z += old[pt[i]]
        return z

    plaintext = toBinary(input[1])

    def enc(final_plain_txt, kv):
        c = 1
        for i in range(1, 17):

            k = kv

            lpl = final_plain_txt[: 64 // 2]
            rpl = final_plain_txt[64 // 2 :]
            expan = permutation(48, rpl, expansion_table)
            ans = ""
            ans2 = ""
            ans3 = ""
            for j in range(0, 48):
                ans += str(int(k[i - 1][j]) ^ int(expan[j]))

            # ans = textwrap.wrap(ans, 6)
            ans = [ans[j : j + 6] for j in range(0, len(ans), 6)]
            for i in range(8):
                ans2 += format(sbox(sboxs[i], i, ans), "04b")
            ans2 = permutation(32, ans2, p_sbox)
            for j in range(0, 32):
                ans3 += str(int(lpl[j]) ^ int(ans2[j]))
            lpl = rpl
            rpl = ans3
            final_plain_txt = lpl + rpl
            c += 1

        return permutation(64, rpl + lpl, fp)

    def ksh():
        z = permutation(56, key, pc1)
        finalk = []
        for i in range(1, 17):

            l = z[: 56 // 2]
            r = z[56 // 2 :]

            if i == 1 or i == 2 or i == 9 or i == 16:
                l = l[1:] + l[0:1]
                r = r[1:] + r[0:1]

            else:
                l = l[2:] + l[0:2]
                r = r[2:] + r[0:2]

            z = l + r
            finalk.append(permutation(48, z, pc2))

        return finalk

    while True:
        key = input[0]

        if len(key) == 16:
            break
    key = toBinary(key)

    if len(plaintext) > 64:
        last = ""
        final_plain_txt = [plaintext[i : i + 64] for i in range(0, len(plaintext), 64)]
        zzz = len(final_plain_txt) - 1
        while len(final_plain_txt[zzz]) < 64:
            final_plain_txt[zzz] = f'{"0"}{final_plain_txt[zzz]}'

        for i in range(0, zzz + 1):
            n = ""
            last += final_plain_txt[i]
            final_plain_txt[i] = permutation(64, final_plain_txt[i], ip)
            n = enc(final_plain_txt[i], ksh())
            res += n
    else:
        while len(plaintext) < 64:
            plaintext = f'{"0"}{plaintext}'
        final_plain_txt = plaintext
        final_plain_txt = permutation(64, final_plain_txt, ip)
        res = enc(final_plain_txt, ksh())

    return hex(int(res, 2))[2:].upper()


def solve_problem_solving_easy(input) -> list:
    """
    This function takes a tuple as input and returns a list as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A list of strings representing a question.
        - An integer representing a key.

    Returns:
    list: A list of strings representing the solution to the problem.
    """
    strings, x = input

    # Count the occurrences of each string
    frq = Counter(strings)

    # Use a min heap to maintain the top x elements based on frequency and lexicographical order
    min_heap = [(-freq, word) for word, freq in frq.items()]
    heapq.heapify(min_heap)

    ans = []
    while len(ans) < x and min_heap:
        freq, word = heapq.heappop(min_heap)
        ans.append(word)

    return ans


def solve_problem_solving_medium(input: str) -> str:
    """
    This function takes a string as input and returns a string as output.

    Parameters:
    input (str): A string representing the input data.

    Returns:
    str: A string representing the solution to the problem.
    """
    s = input
    counts = []
    result = []
    current = ""
    count = 0

    for i in range(len(s)):
        if "0" <= s[i] <= "9":
            count = count * 10 + int(s[i])
        elif "a" <= s[i] <= "z":
            current += s[i]
        elif s[i] == "[":
            counts.append(count)
            result.append(current)
            current = ""
            count = 0
        elif s[i] == "]":
            repeat = counts.pop()
            prev = result.pop()

            for _ in range(repeat):
                prev += current
            current = prev

    return current


def solve_problem_solving_hard(input: tuple) -> int:
    """
    This function takes a tuple as input and returns an integer as output.

    Parameters:
    input (tuple): A tuple containing two integers representing m and n.

    Returns:
    int: An integer representing the solution to the problem.
    """
    n, m = input

    dp = [[0] * (m + 5) for _ in range(n + 5)]
    dp[n - 1][m - 1] = 1

    for j in range(m - 2, -1, -1):
        dp[n - 1][j] = dp[n - 1][j + 1]

    for i in range(n - 2, -1, -1):
        dp[i][m - 1] = dp[i + 1][m - 1]

    for i in range(n - 2, -1, -1):
        for j in range(m - 2, -1, -1):
            dp[i][j] = dp[i + 1][j] + dp[i][j + 1]

    return dp[0][0]


riddle_solvers = {
    "cv_easy": solve_cv_easy,
    "cv_medium": solve_cv_medium,
    "cv_hard": solve_cv_hard,
    "ml_easy": solve_ml_easy,
    "ml_medium": solve_ml_medium,
    "sec_medium_stegano": solve_sec_medium,
    "sec_hard": solve_sec_hard,
    "problem_solving_easy": solve_problem_solving_easy,
    "problem_solving_medium": solve_problem_solving_medium,
    "problem_solving_hard": solve_problem_solving_hard,
}


# image_array = np.random.randint(0, 256, size=(400, 360, 3), dtype=np.uint8)
# image_array = image_array.astype(np.float32)

# print(solve_sec_medium(image_array.tolist()))


# inp = [
#     [
#         [13, 11, 33],
#         [14, 15, 36],
#         [12, 13, 33],
#         [12, 17, 36],
#         [20, 26, 48],
#         [22, 32, 57],
#         [27, 36, 67],
#         [34, 44, 80],
#         [51, 62, 107],
#         [48, 59, 104],
#         [55, 62, 108],
#         [42, 47, 85],
#         [35, 40, 70],
#         [37, 42, 61],
#         [23, 25, 37],
#         [17, 21, 30],
#         [20, 23, 40],
#         [19, 24, 44],
#         [19, 24, 44],
#         [20, 25, 45],
#     ],
#     [
#         [11, 9, 30],
#         [13, 14, 34],
#         [13, 14, 34],
#         [11, 16, 35],
#         [17, 24, 43],
#         [19, 30, 52],
#         [28, 37, 68],
#         [40, 50, 86],
#         [44, 56, 98],
#         [42, 53, 98],
#         [51, 60, 103],
#         [40, 48, 85],
#         [29, 34, 63],
#         [23, 28, 48],
#         [17, 21, 32],
#         [26, 30, 39],
#         [18, 24, 40],
#         [18, 23, 43],
#         [19, 24, 44],
#         [20, 25, 45],
#     ],
#     [
#         [10, 9, 27],
#         [13, 14, 32],
#         [14, 15, 33],
#         [11, 17, 33],
#         [16, 23, 41],
#         [18, 29, 49],
#         [30, 39, 68],
#         [46, 56, 92],
#         [57, 69, 111],
#         [49, 60, 105],
#         [51, 60, 103],
#         [40, 48, 85],
#         [26, 33, 61],
#         [16, 23, 42],
#         [11, 20, 29],
#         [25, 32, 40],
#         [19, 25, 41],
#         [19, 24, 43],
#         [20, 25, 44],
#         [21, 26, 45],
#     ],
#     [
#         [14, 13, 29],
#         [16, 15, 31],
#         [13, 15, 30],
#         [11, 17, 31],
#         [16, 23, 39],
#         [20, 28, 49],
#         [32, 41, 70],
#         [49, 59, 94],
#         [74, 86, 126],
#         [59, 70, 115],
#         [45, 57, 99],
#         [35, 45, 81],
#         [27, 36, 65],
#         [20, 31, 49],
#         [17, 27, 37],
#         [18, 28, 37],
#         [19, 26, 42],
#         [21, 26, 45],
#         [22, 27, 46],
#         [22, 27, 46],
#     ],
#     [
#         [18, 16, 30],
#         [16, 15, 29],
#         [10, 12, 25],
#         [12, 16, 28],
#         [17, 23, 39],
#         [20, 28, 47],
#         [29, 39, 66],
#         [46, 56, 91],
#         [72, 84, 124],
#         [62, 75, 119],
#         [43, 56, 100],
#         [31, 43, 81],
#         [26, 40, 69],
#         [27, 39, 61],
#         [25, 39, 52],
#         [18, 31, 40],
#         [20, 29, 44],
#         [21, 28, 44],
#         [22, 29, 45],
#         [22, 29, 45],
#     ],
#     [
#         [17, 15, 29],
#         [14, 12, 26],
#         [10, 9, 23],
#         [12, 14, 27],
#         [16, 22, 38],
#         [18, 25, 44],
#         [25, 35, 62],
#         [42, 52, 87],
#         [67, 79, 121],
#         [72, 85, 130],
#         [60, 73, 118],
#         [45, 58, 100],
#         [32, 47, 80],
#         [25, 41, 66],
#         [27, 44, 60],
#         [21, 37, 50],
#         [22, 31, 46],
#         [23, 30, 46],
#         [24, 31, 47],
#         [25, 32, 48],
#     ],
#     [
#         [13, 10, 27],
#         [13, 11, 25],
#         [12, 11, 25],
#         [15, 17, 30],
#         [19, 22, 39],
#         [17, 22, 44],
#         [26, 33, 62],
#         [44, 52, 89],
#         [67, 78, 123],
#         [78, 90, 138],
#         [74, 86, 134],
#         [67, 82, 125],
#         [47, 63, 99],
#         [27, 44, 72],
#         [28, 46, 66],
#         [24, 41, 57],
#         [25, 37, 53],
#         [28, 35, 53],
#         [29, 36, 54],
#         [30, 37, 55],
#     ],
#     [
#         [12, 7, 27],
#         [14, 11, 28],
#         [17, 14, 31],
#         [21, 20, 36],
#         [21, 22, 42],
#         [19, 21, 46],
#         [28, 33, 65],
#         [49, 54, 94],
#         [65, 71, 119],
#         [74, 84, 135],
#         [73, 85, 137],
#         [78, 92, 139],
#         [61, 79, 119],
#         [33, 52, 84],
#         [31, 51, 76],
#         [28, 46, 66],
#         [30, 41, 59],
#         [34, 41, 59],
#         [34, 41, 59],
#         [35, 42, 60],
#     ],
#     [
#         [17, 9, 32],
#         [17, 12, 34],
#         [19, 14, 36],
#         [21, 18, 39],
#         [24, 22, 46],
#         [22, 22, 50],
#         [33, 33, 69],
#         [51, 54, 97],
#         [74, 77, 128],
#         [80, 85, 141],
#         [99, 108, 165],
#         [104, 116, 168],
#         [78, 92, 137],
#         [51, 69, 107],
#         [39, 60, 91],
#         [34, 52, 76],
#         [36, 47, 67],
#         [38, 45, 64],
#         [41, 48, 67],
#         [41, 48, 67],
#     ],
#     [
#         [25, 17, 41],
#         [24, 16, 40],
#         [22, 14, 37],
#         [22, 17, 40],
#         [26, 22, 47],
#         [26, 23, 54],
#         [35, 33, 72],
#         [53, 51, 98],
#         [76, 76, 130],
#         [80, 82, 139],
#         [102, 107, 165],
#         [128, 137, 192],
#         [105, 119, 168],
#         [68, 86, 126],
#         [40, 60, 93],
#         [42, 59, 85],
#         [43, 54, 76],
#         [45, 52, 71],
#         [48, 55, 74],
#         [50, 57, 76],
#     ],
#     [
#         [28, 18, 43],
#         [25, 15, 40],
#         [21, 11, 35],
#         [23, 15, 39],
#         [28, 19, 46],
#         [29, 22, 53],
#         [38, 32, 70],
#         [56, 50, 96],
#         [81, 76, 130],
#         [86, 85, 143],
#         [96, 98, 157],
#         [120, 127, 182],
#         [110, 122, 172],
#         [85, 100, 141],
#         [54, 73, 106],
#         [57, 74, 100],
#         [51, 62, 82],
#         [50, 59, 76],
#         [55, 64, 81],
#         [59, 68, 85],
#     ],
#     [
#         [31, 21, 46],
#         [31, 21, 45],
#         [29, 19, 43],
#         [32, 22, 46],
#         [37, 27, 52],
#         [39, 28, 58],
#         [49, 39, 76],
#         [66, 57, 102],
#         [86, 77, 130],
#         [101, 96, 154],
#         [103, 102, 160],
#         [95, 98, 153],
#         [105, 113, 160],
#         [99, 113, 152],
#         [80, 96, 129],
#         [64, 80, 105],
#         [66, 77, 95],
#         [62, 71, 86],
#         [66, 75, 90],
#         [73, 82, 97],
#     ],
#     [
#         [35, 23, 47],
#         [37, 25, 47],
#         [36, 24, 44],
#         [39, 27, 47],
#         [41, 29, 51],
#         [40, 27, 55],
#         [49, 35, 70],
#         [66, 53, 96],
#         [88, 76, 126],
#         [112, 103, 158],
#         [122, 119, 174],
#         [96, 98, 149],
#         [121, 128, 172],
#         [124, 134, 170],
#         [110, 125, 154],
#         [77, 92, 113],
#         [80, 92, 106],
#         [71, 81, 90],
#         [72, 82, 91],
#         [81, 91, 100],
#     ],
#     [
#         [44, 32, 52],
#         [45, 33, 53],
#         [44, 31, 49],
#         [46, 33, 51],
#         [48, 35, 55],
#         [45, 31, 56],
#         [53, 36, 68],
#         [68, 52, 91],
#         [89, 76, 122],
#         [109, 99, 149],
#         [127, 120, 172],
#         [110, 108, 157],
#         [134, 137, 178],
#         [137, 146, 179],
#         [130, 144, 170],
#         [99, 115, 131],
#         [70, 83, 91],
#         [58, 69, 73],
#         [57, 68, 72],
#         [67, 78, 82],
#     ],
#     [
#         [53, 40, 58],
#         [50, 37, 54],
#         [47, 33, 48],
#         [52, 38, 53],
#         [61, 47, 64],
#         [63, 48, 69],
#         [71, 52, 80],
#         [83, 66, 102],
#         [97, 82, 125],
#         [115, 103, 149],
#         [125, 117, 166],
#         [120, 117, 162],
#         [117, 119, 157],
#         [128, 135, 164],
#         [115, 127, 149],
#         [88, 102, 115],
#         [46, 60, 63],
#         [32, 44, 42],
#         [33, 45, 43],
#         [48, 60, 58],
#     ],
#     [
#         [36, 25, 39],
#         [30, 19, 33],
#         [25, 15, 26],
#         [36, 24, 36],
#         [53, 41, 53],
#         [61, 48, 65],
#         [69, 56, 76],
#         [81, 67, 93],
#         [104, 90, 123],
#         [131, 121, 156],
#         [134, 126, 163],
#         [132, 128, 161],
#         [98, 99, 127],
#         [113, 118, 140],
#         [82, 89, 107],
#         [46, 56, 65],
#         [35, 45, 46],
#         [20, 31, 27],
#         [25, 34, 31],
#         [43, 54, 50],
#     ],
#     [
#         [12, 9, 18],
#         [1, 0, 7],
#         [2, 0, 6],
#         [15, 10, 17],
#         [18, 13, 20],
#         [19, 13, 23],
#         [41, 35, 47],
#         [72, 66, 80],
#         [58, 54, 69],
#         [96, 92, 107],
#         [122, 119, 136],
#         [88, 86, 100],
#         [41, 40, 54],
#         [20, 20, 32],
#         [6, 9, 18],
#         [6, 9, 14],
#         [14, 18, 19],
#         [14, 19, 15],
#         [17, 19, 16],
#         [15, 20, 16],
#     ],
#     [
#         [0, 0, 5],
#         [0, 0, 5],
#         [0, 0, 5],
#         [0, 0, 5],
#         [0, 0, 5],
#         [0, 0, 5],
#         [5, 4, 10],
#         [15, 14, 20],
#         [0, 0, 5],
#         [2, 1, 7],
#         [14, 13, 19],
#         [8, 7, 13],
#         [5, 4, 10],
#         [7, 6, 12],
#         [0, 0, 5],
#         [0, 0, 4],
#         [0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0],
#     ],
#     [
#         [0, 0, 4],
#         [4, 3, 8],
#         [4, 3, 8],
#         [2, 1, 6],
#         [3, 2, 7],
#         [5, 4, 9],
#         [1, 0, 5],
#         [0, 0, 4],
#         [7, 6, 11],
#         [0, 0, 4],
#         [0, 0, 4],
#         [1, 0, 5],
#         [5, 4, 9],
#         [7, 6, 11],
#         [1, 0, 5],
#         [6, 6, 8],
#         [0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0],
#         [1, 1, 0],
#     ],
#     [
#         [0, 0, 4],
#         [3, 2, 7],
#         [2, 1, 6],
#         [0, 0, 4],
#         [2, 1, 6],
#         [7, 6, 11],
#         [6, 5, 10],
#         [0, 0, 4],
#         [2, 1, 6],
#         [0, 0, 4],
#         [2, 1, 6],
#         [1, 0, 5],
#         [0, 0, 4],
#         [0, 0, 4],
#         [0, 0, 4],
#         [1, 1, 3],
#         [2, 2, 2],
#         [2, 2, 0],
#         [2, 2, 0],
#         [2, 2, 0],
#     ],
# ]
# print(np.array(inp).shape)
