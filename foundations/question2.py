import sys

def ComputeMinCost(n, c):
    dp = [sys.maxint for i in range(n)]
    dp[0] = 0

    for i in range(n):
        pre = dp[0]
        for j in range(n):
            # cost(i,j) = min (move_from_top, move_from_left) + c(i,j)
            # dp array is used for caching cost of cells in above row.
            # pre is used for caching cost of the left cell
            dp[j] = min(dp[j], pre) + c(i,j)
            pre = dp[j]

    return dp[n-1]

def ClimbStairs(n):
    dp = [0 for i in range(n)]

    for i in range(n):
        dp[i] = 1
        for j in range(i):
            dp[i] += dp[i-j]

    return dp[n-1]