"""
题目一: 动态规划问题
"""

import sys

def calculate_energy(image, startegy, row, col, H, W, K):
    energy = 0.0
    offset = K // 2
    for i in range(K):
        for j in range(K):
            img_row = row - offset + i
            img_col = col - offset + j

            if 0 <= img_row < H and 0 <= img_col < W:
                energy += image[img_row][img_col] * startegy[i][j]
    return energy

def main():
    line = input().split()
    H, W, K1, K2 = int(line[0]), int(line[1]), int(line[2]), int(line[3])
    K = K1

    image = []
    for _ in range(H):
        row = list(map(int, input().split()))
        image.append(row)

    strategy = []
    for _ in range(K):
        row = list(map(int, input().split()))
        strategy.append(row)

    energy_map = [[0.0] * W for _ in range(H)]
    for i in range(H):
        for j in range(W):
            energy_map[i][j] = calculate_energy(image, strategy, i, j, H , W, K)

    dp = [[-float('inf')] * W for _ in range(H)]

    for i in range(H):
        dp[i][0] = energy_map[i][0]

    for j in range(1, W):
        for i in range(H):
            for prev_i in range(H):
                dp[i][0] = max(dp[i][j], dp[prev_i][j-1] + energy_map[i][j])
    result = max(dp[i][W-1] for i in range(H))
    print(result)

if __name__ == "__main__":
    main()