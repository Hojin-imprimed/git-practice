from functools import cache
from typing import List
from collections import deque


class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        k = 1
        for i in range(1, len(nums)):
            if nums[k] is None or nums[k] < nums[i]:
                nums[k] = nums[i]
                k += 1
            else:
                nums[i] = None
        return nums

    def reverse(self, x: int) -> int:
        is_negative = x < 0
        if is_negative:
            str_x = str(-x)[::-1]
            if len(str_x) >= 10 and str_x > '2147483648':
                return 0
        else:
            str_x = str(x)[::-1]
            if len(str_x) >= 10 and str_x > '2147483647':
                return 0
        return -int(str_x) if is_negative else int(str_x)

    def uniquePaths(self, m: int, n: int) -> int:
        res = [[1] * n for _ in range(m)]
        print(res)
        for i in range(1, m):
            for j in range(1, n):
                res[i][j] = res[i][j - 1] + res[i - 1][j]

        return res[-1][-1]

    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        row = len(obstacleGrid)
        col = len(obstacleGrid[0])
        dp = obstacleGrid.copy()
        for i in range(row):
            if obstacleGrid[i][0] == 1:
                break
            dp[i][0] = 1
        for j in range(1, col):
            if obstacleGrid[0][j] == 1:
                break
            dp[0][j] = 1

        for i in range(1, row):
            for j in range(1, col):
                if obstacleGrid[i][j] == 1:
                    dp[i][j] = 0
                else:
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]

    def minDistance(self, word1: str, word2: str) -> int:
        @cache
        def dp(i, j):
            if i == len(word1):
                return len(word2) - j
            if j == len(word2):
                return len(word1) - i

            if word1[i] == word2[j]:
                return dp(i + 1, j + 1, word1, word2)
            else:
                insert = dp(i, j + 1, word1, word2)
                delete = dp(i + 1, j, word1, word2)
                replace = dp(i + 1, j + 1, word1, word2)
                return 1 + min(insert, delete, replace)

        return dp(0, 0)

    def numIslands(self, grid: List[List[str]]) -> int:
        def remove_adjacent_1s(i, j):
            if -1 < i < len(grid) and -1 < j < len(grid[0]) and grid[i][j] == '1':
                grid[i][j] = 0
                remove_adjacent_1s(i, j + 1)
                remove_adjacent_1s(i, j - 1)
                remove_adjacent_1s(i + 1, j)

        num = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '1':
                    num += 1
                    remove_adjacent_1s(i, j)
        return num

    def numIslands2(self, grid: List[List[str]]) -> int:
        def bfs(q):
            while q:
                i, j = q.popleft()
                if grid[i][j] == '1':
                    grid[i][j] = '0'
                    for a, b in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                        c = i + a
                        d = j + b
                        if -1 < c < len(grid) and -1 < d < len(grid[0]):
                            q.append((c, d))

        num = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '1':
                    num += 1
                    bfs(deque((i, j)))

        return num

    def climbStairs(self, n: int) -> int:
        def fn(step):
            if step == n:
                return 1
            return fn(step + 1) + fn(step + 2)

        return fn(1) + fn(2)

    def climbStairs2(self, n: int) -> int:
        if n < 3:
            return n

        dp = [0] * (n + 1)
        dp[1] = 1
        dp[2] = 2
        for i in range(3, n + 1):
            dp[i] = dp[i - 2] + dp[i - 1]
            dp[2] = dp[0] + dp[1]
        return dp[n]


class MinStack:
    def __init__(self):
        self.head = None

    def push(self, val: int) -> None:
        if self.head is None:
            self.head = Node(val, val)
        else:
            self.head = Node(val, min(val, self.head.min), self.head)

    def pop(self) -> None:
        self.head = self.head.next

    def top(self) -> int:
        return self.head.val

    def getMin(self) -> int:
        return self.head.min


class Node:
    def __init__(self, val, min, next=None):
        self.val = val
        self.min = min
        self.next = next


def addBinary(a: str, b: str) -> str:
    x, y = int(a, 2), int(b, 2)
    while y:
        answer = x ^ y
        carry = (x & y) << 1
        x, y = answer, carry
    return bin(x)[2:]


def merge(a: List[int], m: int, b: List[int], n: int) -> None:
    while m > 0 and n > 0:
        if a[m - 1] >= b[n - 1]:
            a[m + n - 1] = a[m - 1]
            m -= 1
        elif a[m - 1] < b[n - 1]:
            a[m + n - 1] = b[n - 1]
            n -= 1
    if n > 0:
        a[:n] = b[:n]
    else:
        pass
    return a

from collections import defaultdict

class TrieNode(object):
    def __init__(self, value=-1):
        self.children = {}
        self.value = value


class FileSystem:

    def __init__(self):
        self.root = TrieNode()

    def createPath(self, path: str, value: int) -> bool:
        # if the path already exists: false
        # if its parent path doesn't exist: false
        ps = path.split("/")[1:]
        cur = self.root
        for component in ps[:-1]:
            if component not in cur.children:
                return False
            cur = cur.children[component]
        last_path = ps[-1]
        if last_path in cur.children:
            return False
        cur.children[last_path] = TrieNode(value)
        return True

    def get(self, path: str) -> int:
        components = path.split("/")[1:]
        cur = self.root
        for comp in components:
            if comp not in cur.children:
                return -1
            cur = cur.children[comp]
        return cur.value


from collections import Counter


class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        last_occurrence = {char: i for i, char in enumerate(s)}
        print(last_occurrence)
        stack = []

        for i, char in enumerate(s):
            if char not in stack:
                while stack and char < stack[-1] and i < last_occurrence[stack[-1]]:
                    stack.pop()
                stack.append(char)

        return ''.join(stack)

    def winnerOfGame(self, colors: str) -> bool:
        # find more than three pieces that are the same color
        # count of each player's turn how many they need to remove
        # shorter lose
        num = {'A': [], 'B': []}
        cnt = 1
        for i in range(len(colors)-1):
            if colors[i] == colors[i+1]:
                cnt += 1
            else:
                if cnt > 2:
                    num[colors[i]].append(cnt-2)
                cnt = 1
        if cnt > 2:
            num[colors[-1]].append(cnt-2)
        return sum(num['A']) > sum[num['B']]


Solution().winnerOfGame("AAAABBBB")